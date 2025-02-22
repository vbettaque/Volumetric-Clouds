shader_type sky;
render_mode use_half_res_pass;

#define SUNDIR LIGHT0_DIRECTION
#define SUNSIZE LIGHT0_SIZE

const float EPSILON = 0.01;
const float BASE_UVW_BASE_SCALE = 0.00001;
const float DETAIL_UVW_BASE_SCALE = 0.001;
const float DENSITY_BASE_SCALE = 0.01;
const float WEATHER_UV_BASE_SCALE = 0.00001;

const float SUN_ENERGY = 1000.0;

const vec3 RANDOM_VECTORS[6] = {
	vec3( 0.38051305f,  0.92453449f, -0.02111345f),
	vec3(-0.50625799f, -0.03590792f, -0.86163418f),
	vec3(-0.32509218f, -0.94557439f,  0.01428793f),
	vec3( 0.09026238f, -0.27376545f,  0.95755165f),
	vec3( 0.28128598f,  0.42443639f, -0.86065785f),
	vec3(-0.16852403f,  0.14748697f,  0.97460106f)
};

const float FOUR_PI_INV = 0.0795774715459;

uniform ivec2 viewport_size;

uniform float time;

group_uniforms Sky;
uniform vec3 base_color_a: source_color = vec3(0.78, 0.86, 1.);
uniform vec3 base_color_b: source_color = vec3(0.4, 0.6, 0.9);
uniform float rayleigh : hint_range(0, 64) = 2.0;
uniform vec4 rayleigh_color : source_color = vec4(0.26, 0.41, 0.58, 1.0);
uniform float mie : hint_range(0, 1) = 0.005;
uniform float mie_eccentricity : hint_range(-1, 1) = 0.8;
uniform vec4 mie_color : source_color = vec4(0.63, 0.77, 0.92, 1.0);
uniform float turbidity : hint_range(0, 1000) = 10.0;
const float rayleigh_zenith_size = 8.4e3;
const float mie_zenith_size = 1.25e3;
uniform vec4 ground_color : source_color = vec4(1.0);
group_uniforms;

group_uniforms Clouds;
uniform float ground_radius = 6371000.;
uniform float cloud_min_height: hint_range(0, 1500, 10) = 1500.0;
uniform float cloud_max_height: hint_range(1500, 10000, 10) = 4000.0;


uniform vec4 stratus_gradient = vec4(0.02, 0.05, 0.09, 0.11);
uniform vec4 cumulus_gradient = vec4(0.02, 0.2, 0.48, 0.625);
uniform vec4 cumulonimbus_gradient = vec4(0.01, 0.0625, 0.78, 1.0);

uniform float extinction_scale: hint_range(0.0, 1.0, 0.01) = 0.1;
uniform float cloud_coverage: hint_range(0.0, 1.0, 0.01) = 0.25;
uniform float droplet_radius: hint_range(1.0, 50.0, 0.1) = 4; // micrometer
uniform vec3 albedo: source_color = vec3(1.);
uniform float powder_coefficient: hint_range(0.0, 20.0, 0.1) = 2;

group_uniforms Clouds.Base;
uniform float base_uvw_scale: hint_range(0.0, 50.0, 0.1) = 8;
uniform vec3 base_fbm_weights = vec3(0.625, 0.25, 0.125);
uniform float base_wind_scale = 12.;
group_uniforms;

group_uniforms Clouds.Detail;
uniform float erosion_scale: hint_range(0.0, 1.0, 0.1) = 1;
uniform float detail_uvw_scale = 1;
uniform vec3 detail_fbm_weights = vec3(0.625, 0.25, 0.125);
uniform vec3 detail_direction;
uniform float detail_speed = 40.;
group_uniforms;

group_uniforms Clouds.Phase;
uniform float eccentricity_1: hint_range(-1.0, 1.0, 0.1) = -0.4;
uniform float eccentricity_2: hint_range(-1.0, 1.0, 0.1) = 0.7;
uniform float phase_mix_weight: hint_range(0.0, 1.0, 0.1) = 0.5;
group_uniforms;

group_uniforms Clouds.MultiScattering;
uniform int scattering_octaves: hint_range(1, 10, 1) = 8;
uniform float scattering_attenuation: hint_range(0.0, 1.0, 0.1) = 0.5;
uniform float extinction_attenuation: hint_range(0.0, 1.0, 0.1) = 0.5;
uniform float eccentricity_attenuation: hint_range(0.0, 1.0, 0.1) = 0.5;

group_uniforms;

group_uniforms CloudMarch;
uniform int min_steps: hint_range(1, 1000, 1) = 64;
uniform int max_steps: hint_range(1, 1000, 1) = 128;
uniform int substeps: hint_range(0, 20) = 0;
uniform float march_offset_scale: hint_range(0.0, 10.0, 0.1) = 1.;
group_uniforms;

group_uniforms LightMarch;
uniform int light_march_steps: hint_range(1, 100, 1) = 4;
uniform float light_march_absorption = 1.;
uniform float darkness_threshold: hint_range(0.0, 1., 0.01) = 0;
group_uniforms;

group_uniforms Weather;
uniform vec2 wind_direction = vec2(1, 0);
uniform float wind_speed: hint_range(0.0, 20.0) = 1.0;
uniform sampler2D weather_texture: filter_linear_mipmap, repeat_enable;
uniform float weather_uv_scale = 6.;
uniform float weather_wind_scale = 16.;
group_uniforms;

group_uniforms Sun;
uniform float exposure : hint_range(0, 128) = 0.1;
group_uniforms;

group_uniforms Noise;
uniform sampler3D base_noise_texture: filter_linear_mipmap, repeat_enable;
uniform sampler3D detail_noise_texture: filter_linear_mipmap, repeat_enable;
uniform sampler2D blue_noise: repeat_enable;
group_uniforms;


// Position hashing (https://www.shadertoy.com/view/4sfGzS).
// TODO: Replace with something else?
float hash(vec3 p) {
	p = fract(p * 0.3183099 + 0.1);
	p *= 17.;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}


float sum4(vec4 vec) { return vec.x + vec.y + vec.z + vec.w; }
float sum3(vec3 vec) { return vec.x + vec.y + vec.z; }


// Maps value from one range to another.
float remap(
    float value, float old_min, float old_max, float new_min, float new_max
) {
    float scale_factor = (new_max - new_min) / (old_max - old_min);
	return new_min + (value - old_min) * scale_factor;
}


// Henyey-Greenstein phase function.
float henyey_greenstein(float cos_theta, float g) {
	return FOUR_PI_INV * (1. - g * g) /
		(pow(1. + g * g - 2. * g * cos_theta, 1.5));
}


// Linearly interpolates between two Henyey-Greenstein phase functions.
float dual_lobe_hg(float cos_theta, float g_scale) {
	float hg_1 = henyey_greenstein(cos_theta, g_scale * eccentricity_1);
	float hg_2 = henyey_greenstein(cos_theta, g_scale * eccentricity_2);
	return mix(hg_1, hg_2, phase_mix_weight);
}


// Returns distance to the cloud layer given the current view direction.
// TODO: Check for correctness
float dist_to_radius(vec3 pos, vec3 ray_dir, float radius) {
	float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(ray_dir, pos);
    float c = dot(pos, pos) - (radius * radius);
	float d = sqrt((b*b) - 4.0*a*c);
	float p = -b - d;
	float p2 = -b + d;
    return max(p, p2) / (2.0 * a);
}

// Returns relative height within cloud layer.
// TODO: Account for curvature?
float relative_height(vec3 pos) {
	float cloud_min_radius = ground_radius + cloud_min_height;
	float cloud_max_radius = ground_radius + cloud_max_height;
    float rel_height = remap(pos.y, cloud_min_radius, cloud_max_radius, 0., 1.);
	return clamp(rel_height, 0., 1.);
}


// Linearly interpolates between the three different cloud gradients depending
// on the provided cloud type (between 0 and 1).
float cloud_gradient(float rel_height, float cloud_type) {
	float stratus = 1. - clamp(cloud_type * 2., 0., 1.);
	float cumulus = 1. - abs(cloud_type - 0.5) * 2.;
	float cumulonimbus = clamp(cloud_type - 0.5, 0., 1.) * 2.;
	vec4 gradient = stratus * stratus_gradient
        + cumulus * cumulus_gradient
        + cumulonimbus * cumulonimbus_gradient;
	return smoothstep(gradient.x, gradient.y, rel_height)
        - smoothstep(gradient.z, gradient.w, rel_height);
}


vec2 wind_offset() {
    return time * wind_speed * normalize(wind_direction);
}


// Samples from the weather texture.
// TODO: Change time offset to weather offset since it's not used elsewhere?
vec4 weather(vec2 pos) {
    vec2 weather_offset = weather_wind_scale * wind_offset();
	vec2 uv = weather_uv_scale * WEATHER_UV_BASE_SCALE
		* (pos + weather_offset) + 0.5;
	return texture(weather_texture, uv);
}


// Simple Analytic sky. In a real project you should use a texture
vec3 atmosphere(vec3 ray_dir) {
	vec3 dir_to_light = normalize(LIGHT0_DIRECTION);
	float zenith = acos(dir_to_light.y);
	float sun_energy = max(0.0, 1.0 - exp(-((PI * 0.5) - zenith))) * SUN_ENERGY * LIGHT0_ENERGY;
	float sun_fade = 1.0 - clamp(1.0 - exp(LIGHT0_DIRECTION.y), 0.0, 1.0);

	// Rayleigh coefficients.
	float rayleigh_coefficient = rayleigh - ( 1.0 * ( 1.0 - sun_fade ) );
	vec3 rayleigh_beta = rayleigh_coefficient * rayleigh_color.rgb * 0.0001;
	// mie coefficients from Preetham
	vec3 mie_beta = turbidity * mie * mie_color.rgb * 0.000434;

	// optical length
	float optical_zenith = acos(max(0.0, ray_dir.y));
	float optical_mass = 1.0 / (cos(optical_zenith) + 0.15 * pow(93.885 - degrees(optical_zenith), -1.253));
	float rayleigh_scatter = rayleigh_zenith_size * optical_mass;
	float mie_scatter = mie_zenith_size * optical_mass;

	// light extinction based on thickness of atmosphere
	vec3 extinction = exp(-(rayleigh_beta * rayleigh_scatter + mie_beta * mie_scatter));

	// in scattering
	float cos_theta = dot(ray_dir, dir_to_light);

	float rayleigh_phase = (3.0 / (16.0 * PI)) * (1.0 + pow(cos_theta * 0.5 + 0.5, 2.0));
	vec3 betaRTheta = rayleigh_beta * rayleigh_phase;

	float mie_phase = henyey_greenstein(cos_theta, mie_eccentricity);
	vec3 betaMTheta = mie_beta * mie_phase;

	vec3 Lin = pow(sun_energy * ((betaRTheta + betaMTheta) / (rayleigh_beta + mie_beta)) * (1.0 - extinction), vec3(1.5));
	// Hack from https://github.com/mrdoob/three.js/blob/master/examples/jsm/objects/Sky.js
	Lin *= mix(vec3(1.0), pow(sun_energy * ((betaRTheta + betaMTheta) / (rayleigh_beta + mie_beta)) * extinction, vec3(0.5)), clamp(pow(1.0 - zenith, 5.0), 0.0, 1.0));

	// Hack in the ground color
	Lin  *= mix(ground_color.rgb, vec3(1.0), smoothstep(-0.1, 0.1, ray_dir.y));

	// Solar disk and out-scattering
	float sunAngularDiameterCos = cos(SUNSIZE);
	float sunAngularDiameterCos2 = cos(SUNSIZE*0.5);
	float sundisk = smoothstep(sunAngularDiameterCos, sunAngularDiameterCos2, cos_theta);
	vec3 L0 = (sun_energy * 1900.0 * extinction) * sundisk * LIGHT0_COLOR;
	// Note: Add nightime here: L0 += night_sky * extinction

	vec3 color = (Lin + L0) * 0.04;
	color = pow(color, vec3(1.0 / (1.2 + (1.2 * sun_fade))));
	color *= exposure;
	return color;
}


// Samples the cloud density (between 0 and 1).
float cloud_density(vec3 pos, bool has_detail, float lod) {
    float rel_height = relative_height(pos);
    vec4 weather = weather(pos.xz);
	float coverage = cloud_coverage * weather.b;
	float gradient = cloud_gradient(rel_height, weather.r);

	if (gradient < EPSILON || coverage < EPSILON) return 0.;

    vec3 base_offset = vec3(0.);
    base_offset.xz = base_wind_scale * wind_offset();

	vec3 base_uvw = base_uvw_scale * BASE_UVW_BASE_SCALE * (pos + base_offset);
	vec4 base_noise = textureLod(base_noise_texture, base_uvw, lod - 2.);
	float base_fbm = dot(base_noise.gba, base_fbm_weights);
	float base_density = remap(base_noise.r, -(1. - base_fbm), 1., 0., 1.); // TODO: Rethink validity
	base_density = remap(base_density * gradient, 1. - coverage, 1., 0., 1.);
	base_density *= coverage; // * clamp(4.*rel_height, 0, 1) ?

	if (base_density < EPSILON) return 0.;
	if (!has_detail) return base_density;

    vec3 detail_offset = vec3(0., 1., 0.);
    detail_offset.xz = normalize(wind_direction);
    detail_offset = base_density
        - normalize(detail_offset) * time * detail_speed;
	vec3 detail_uvw = detail_uvw_scale * DETAIL_UVW_BASE_SCALE
		* (pos + detail_offset);
	vec3 detail_noise = textureLod(detail_noise_texture, detail_uvw, lod).gba;
	float detail_fbm = dot(detail_noise, detail_fbm_weights);

	// Makes details "fluffy" at the bottom and "wispy" at the top.
	detail_fbm = mix(1. - detail_fbm, detail_fbm,
		clamp(10. * rel_height, 0., 1.)); // TODO: Identify scale paramter

	// Subtracts detail from base edges and renormalizes density to [0, 1].
	float density = remap(base_density,
		0.2 * detail_fbm * (1. - base_density) * rel_height, 1., 0., 1.); // TODO: Identify scale paramter

	// Reduces density at the bottom and increases density at the top.
	density = pow(clamp(density, 0, 1), (1. - rel_height) * 0.8 + 0.5); // TODO: Rewrite as lerp and identify relevant scales. Apply for base shape already?
	return clamp(density, 0, 1);
}


float beer(float extinct, float step_size) {
	return exp(-extinct * step_size);
}


// Returns Beer term integrated over a single step.
// Assumes radiance and extinction are slowly varying over step size.
// Note: Result is not divided by extinction coefficient as it later cancels
// with the scattering coefficient (which is about equal since albedo ~ 1).
float int_beer(float init_transmit, float extinct, float step_size) {
	return init_transmit * (1. - beer(extinct, step_size));
}

// Returns powder-corrected Beer term integrated over a single step.
// Assumes radiance and extinction are slowly varying over step size.
// Note: Result is not divided by extinction coefficient as it later cancels
// with the scattering coefficient (which is about equal since albedo ~ 1).
float int_beer_powder(float init_transmit, float extinct, float step_size) {
	float beer = int_beer(init_transmit, extinct, step_size);
	float init_transmit_powder = pow(init_transmit, powder_coefficient + 1.);
	float extinct_powder = (powder_coefficient + 1.) * extinct;
	float powder = int_beer(init_transmit_powder, extinct_powder, step_size)
		/ (powder_coefficient + 1.);
	return beer - powder;
}


// TODO: Banding at low sun angles
float sun_transmittance(vec3 pos) {
    const int STEPS = 6;
    const float STEPS_SQ = float(STEPS * STEPS);
    float step_size = (cloud_max_height - cloud_min_height) / STEPS_SQ;
	float total_density = 0.;
	vec3 sample_pos = pos;
	for (int i = 0; i < STEPS; i++) {
		sample_pos += step_size
            * (SUNDIR + float(i) * RANDOM_VECTORS[i]);
		total_density += cloud_density(sample_pos, true, float(i));
	}
	sample_pos = pos + STEPS_SQ * step_size * SUNDIR / 2.;
	float rel_height = relative_height(sample_pos);
	total_density += cloud_density(sample_pos, true, 5.);
	float extinction = extinction_scale * total_density;
	return beer(extinction, step_size);
}


vec3 scatter_radiance(vec3 pos, float cos_theta) {
	vec3 radiance = atmosphere(SUNDIR) * LIGHT0_ENERGY;
	float transmittance = sun_transmittance(pos);
	float phase = dual_lobe_hg(cos_theta, 1.);
	float multi_scatter = transmittance * phase;
	float a = 1.; float b = 1.;
	for (int k = 1; k < scattering_octaves; k++) {
		transmittance = pow(transmittance, extinction_attenuation);
		a *= scattering_attenuation;
		b *= eccentricity_attenuation;

		phase = dual_lobe_hg(cos_theta, b);
		multi_scatter += a * transmittance * phase;
	}
	return multi_scatter * radiance;
}


vec3 ambient_radiance(vec3 pos) {
	float rel_height = relative_height(pos);
	vec3 atmosphere_ambient = atmosphere(normalize(vec3(1.0, 1.0, 0.0)));
	vec3 atmosphere_ground = atmosphere(normalize(vec3(1.0, -1.0, 0.0)));
	return mix(atmosphere_ground, atmosphere_ambient,
		smoothstep(0, 1, rel_height));
}


bool detail_march(
	vec3 march_start, vec3 march_step, int steps, inout vec3 radiance,
	inout float transmit
) {
	float step_size = length(march_step);
	vec3 march_dir = march_step / step_size;
	float cos_theta = dot(SUNDIR, march_dir);

	vec3 march_pos = march_start - march_step;
	bool has_density = false;

	for(int i = 0; i < steps; i++) {
		march_pos += march_step;
		float density = cloud_density(march_pos, true, 0.);

		if (density <= 0.) continue;
		has_density = true;

		float extinction = extinction_scale * density;

		vec3 ambient = ambient_radiance(march_pos);
		vec3 scatter = scatter_radiance(march_pos, cos_theta);
		float beer_powder = int_beer_powder(transmit, extinction, step_size);

		radiance += beer_powder * (ambient + scatter) * albedo;
		transmit *= beer(extinction, step_size);
		if (transmit < EPSILON) break;
	}

	return has_density;
}


vec4 cloud_march(vec3 march_start, vec3 march_step, int steps) {
	vec3 radiance = vec3(0.);
	float transmittance = 1.;

	if (substeps <= 0) {
		detail_march(march_start, march_step, steps, radiance, transmittance);
		return vec4(radiance, 1. - transmittance);
	}

	float step_size = length(march_step);
	vec3 march_dir = march_step / step_size;
	vec3 march_pos = march_start - march_step;

	float substep_size = step_size / float(substeps);
	vec3 march_substep = substep_size * march_dir;

	bool using_substeps = false;

	for(int i = 0; i < steps; i++) {
		march_pos += march_step;
		float density = cloud_density(march_pos, false, 0.);

		if (density > 0. && !using_substeps) {
			detail_march(
				march_pos - march_step, march_substep, substeps,
				radiance, transmittance
			);
			using_substeps = true;
		}

		if (using_substeps){
			using_substeps = detail_march(
				march_pos, march_substep, substeps, radiance, transmittance
			);
		}

		if (transmittance < EPSILON) break;
	}

	return vec4(radiance, 1. - transmittance);
}


void sky() {
	vec3 ray_dir = normalize(EYEDIR);
	vec2 screen_uv = SCREEN_UV;

	vec3 light_pos = SUNDIR;
	float sun_cos_angle = dot(ray_dir, light_pos);

	if (AT_HALF_RES_PASS) {
		if (ray_dir.y > 0.) {
			vec3 pos = vec3(0, ground_radius, 0);
			float cloud_min_radius = ground_radius + cloud_min_height;
			float cloud_max_radius = ground_radius + cloud_max_height;
			float dist_to_entry = dist_to_radius(pos, ray_dir, cloud_min_radius);
			float dist_to_exit = dist_to_radius(pos, ray_dir, cloud_max_radius);
			float march_dist = dist_to_exit - dist_to_entry;

			// Decrease number of big march steps towards the horizon.
			float steps = mix(float(min_steps), float(max_steps), ray_dir.y);
			vec3 march_step = ray_dir * march_dist / steps;
			vec3 march_start = pos + dist_to_entry * ray_dir + march_step * (hash(10. * pos) - 0.5); // TODO: Add noise to offset

			vec4 cloud_data = cloud_march(march_start, march_step, int(steps));
			COLOR = cloud_data.rgb;
			ALPHA = cloud_data.a;
		} else {
			COLOR = vec3(0.);
			ALPHA = 1.;
		}
	} else {
		float transmittance = HALF_RES_COLOR.a;
		vec3 light_energy = HALF_RES_COLOR.rgb;

		float sky_col_weight = sqrt(max(ray_dir.y, 0));
		vec3 sky_col_base = mix(base_color_a, base_color_b, sky_col_weight);
		sky_col_base = atmosphere(ray_dir);
		vec3 cloud_color = light_energy;
		COLOR = (1. - transmittance) * sky_col_base + cloud_color;
		COLOR = mix(COLOR, sky_col_base, smoothstep(0.6, 1.0, 1.0-ray_dir.y));

	}
}
