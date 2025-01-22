shader_type sky;
render_mode use_half_res_pass;

#define SUNDIR LIGHT0_DIRECTION
#define SUNSIZE LIGHT0_SIZE

const float EPSILON = 0.01;
const float SHAPE_BASE_SCALE = 0.00001;
const float DETAIL_BASE_SCALE = 0.001;
const float DENSITY_BASE_SCALE = 0.01;
const float WEATHER_BASE_SCALE = 0.00001;

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
uniform float ground_radius = 6378000.0;
uniform float cloud_min_height: hint_range(0, 1500, 10) = 1500.0;
uniform float cloud_max_height: hint_range(1500, 10000, 10) = 4000.0;

uniform vec4 stratus_gradient = vec4(0.02, 0.05, 0.09, 0.11);
uniform vec4 cumulus_gradient = vec4(0.02, 0.2, 0.48, 0.625);
uniform vec4 cumulonimbus_gradient = vec4(0.01, 0.0625, 0.78, 1.0);

uniform float density_multiplier: hint_range(0.0, 10.0, 0.1) = 5.;
uniform float cloud_coverage: hint_range(0.0, 1.0, 0.01) = 0.25;
uniform float droplet_radius: hint_range(1.0, 50.0, 0.1) = 4; // micrometer
uniform vec3 color_extinction: source_color = vec3(0.);
uniform float absorbtion_factor: hint_range(0.0, 1.0, 0.1) = 1;
uniform vec3 color_absorption: source_color = vec3(0.);
uniform float powder_strength: hint_range(0.0, 20.0, 0.1) = 2;

group_uniforms Clouds.Shape;
uniform float shape_scale:hint_range(0.0, 50.0, 0.1) = 8;
uniform vec3 shape_noise_weights = vec3(0.625, 0.25, 0.125);
uniform vec3 shape_offset;
group_uniforms;

group_uniforms Clouds.Detail;
uniform float erosion_scale: hint_range(0.0, 1.0, 0.1) = 1;
uniform float detail_scale = 1;
uniform vec3 detail_noise_weights = vec3(0.625, 0.25, 0.125);
uniform vec3 detail_offset;
uniform vec3 detail_direction;
uniform float detail_speed;
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
uniform int min_big_steps: hint_range(1, 1000, 1) = 64;
uniform int max_big_steps: hint_range(1, 1000, 1) = 128;
uniform float small_step_ratio: hint_range(0.1, 1., 0.01) = 1.;
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
uniform sampler2D weather_map: filter_linear_mipmap, repeat_enable;
uniform float weather_scale = 6.;
uniform vec2 weather_offset;
group_uniforms;

group_uniforms Sun;
uniform float exposure : hint_range(0, 128) = 0.1;
group_uniforms;

group_uniforms Noise;
uniform sampler3D perlin_worley_128: filter_linear_mipmap, repeat_enable;
uniform sampler3D worley_32: filter_linear_mipmap, repeat_enable;
uniform sampler2D blue_noise: repeat_enable;
group_uniforms;


float hash(vec3 p) {
	p  = fract( p * 0.3183099 + 0.1 );
	p *= 17.0;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}


float sum4(vec4 vec) { return vec.x + vec.y + vec.z + vec.w; }
float sum3(vec3 vec) { return vec.x + vec.y + vec.z; }


// Utility function that maps a value from one range to another.
float remap(float value,  float old_min,  float old_max,  float new_min,  float new_max) {
	return new_min + (((value - old_min) / (old_max - old_min)) * (new_max - new_min));
}


float henyey_greenstein(float cos_theta, float g) {
	return FOUR_PI_INV * (1.0 - g * g) / 
		(pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5));
}


float dual_lobe_hg(float cos_theta, float g_1, float g_2) {
	float hg_1 = henyey_greenstein(cos_theta, g_1);
	float hg_2 = henyey_greenstein(cos_theta, g_2);
	return mix(hg_1, hg_2, phase_mix_weight);
}


// Returns distance to the cloud layer given the current view direction.
// TODO: Check for correctness
float dst_to_radius(vec3 pos, vec3 ray_dir, float radius) {
	float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(ray_dir, pos);
    float c = dot(pos, pos) - (radius * radius);
	float d = sqrt((b*b) - 4.0*a*c);
	float p = -b - d;
	float p2 = -b + d;
    return max(p, p2) / (2.0 * a);
}

// Return relative ehight within cloud layer
// TODO: Account for curvature?
float relative_height(vec3 pos) {
	float height_diff = abs(cloud_max_height - cloud_min_height);
	float cloud_min_radius = ground_radius + cloud_min_height;
	float rel_height = (pos.y - cloud_min_radius) / height_diff;
	return clamp(rel_height, 0., 1.);
}


float cloud_gradient(float rel_height, float cloud_type) {
	vec2 gradient_uv = vec2(rel_height, 0);
	float stratus = 1. - clamp(cloud_type * 2., 0, 1);
	float cumulus = 1. - abs(cloud_type - 0.5) * 2.;
	float cumulonimbus = clamp(cloud_type - 0.5, 0, 1) * 2.;
	vec4 gradient = stratus * stratus_gradient + cumulus * cumulus_gradient + cumulonimbus * cumulonimbus_gradient;
	return smoothstep(gradient.x, gradient.y, rel_height) - smoothstep(gradient.z, gradient.w, rel_height);
}


vec4 weather(vec2 pos) {
	vec2 uv = pos * weather_scale * WEATHER_BASE_SCALE + 0.5 + weather_offset;
	return texture(weather_map, uv);
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


// [0, 1] (10e6 * C_V)
float concentration(vec3 pos, bool has_detail, float lod) {

	pos.xz += time * 20.0 * normalize(wind_direction) * wind_speed * 0.6;
	float rel_height = relative_height(pos);
	vec4 weather = weather(pos.xz);
	float coverage = cloud_coverage * weather.b;
	float gradient = cloud_gradient(rel_height, weather.r);
	if (gradient < EPSILON || coverage < EPSILON) return 0.;

	vec3 base_uvw = shape_scale * SHAPE_BASE_SCALE * pos;
	vec4 base_noise = textureLod(perlin_worley_128, base_uvw, lod-2.);
	vec3 base_weights = shape_noise_weights / sum3(shape_noise_weights);
	float base_fbm = dot(base_noise.gba, base_weights);
	float base_density = remap(base_noise.r, -(1. - base_fbm), 1, 0, 1);
	base_density = remap(base_density * gradient, 1.-coverage, 1, 0, 1);
	base_density *= coverage;

	if (base_density < EPSILON) return 0.;
	if (!has_detail) return base_density;

	pos.xz -= time * normalize(wind_direction) * 20.;
	pos.y -= time * 20.;
	vec3 detail_uvw = detail_scale * DETAIL_BASE_SCALE * pos;
	vec3 detail_noise = textureLod(worley_32, detail_uvw, lod).gba;
	vec3 detail_weights = detail_noise_weights / sum3(detail_noise_weights);
	float detail_fbm = dot(detail_noise, detail_weights);
	// What's going on here?!
	detail_fbm = mix(detail_fbm, 1.-detail_fbm, clamp(4.*rel_height, 0, 1));
	float density = remap(base_density, 0.4 * detail_fbm * rel_height, 1, 0, 1);
	//density = pow(clamp(density, 0, 1), (1. - rel_height) * 0.8 + 0.5);
	return density_multiplier * DENSITY_BASE_SCALE * density;
}

// Extinction coefficient
vec3 extinction(float conc) {
	float sigma_e = conc / droplet_radius;
	return sigma_e * (1. + color_extinction);
}

vec3 absorbtion(float conc) {
	float sigma_a = absorbtion_factor * conc / droplet_radius;
	return sigma_a * color_absorption;
}

// TODO: Banding at low sung angles
vec3 sun_transmittance(vec3 pos) {
	float step_size = (cloud_max_height - cloud_min_height) / 36.;
	float total_conc = 0.;
	vec3 sample_pos = pos;
	for (int i = 0; i < 6; i++) {
		sample_pos += (SUNDIR + float(i) * RANDOM_VECTORS[i]) * step_size;
		total_conc += concentration(sample_pos, true, float(i));
	}
	sample_pos = pos + 18. * step_size * SUNDIR;
	float rel_height = relative_height(sample_pos);
	total_conc += concentration(sample_pos, true, 5.);
	vec3 extinct = extinction(total_conc);
	//vec3 beers = exp(-extinct * step_size);
	//vec3 beers2 = exp(-extinct * step_size * 0.25) * 0.7;
	//return max(beers, beers2);
	return exp(-extinct * step_size);
}


vec4 cloud_march(vec3 pos, vec3 ray_dir, vec2 screen_uv) {
	float dst_to_entry = dst_to_radius(pos, ray_dir, ground_radius + cloud_min_height);
	if (dst_to_entry > 35000.) return vec4(0, 0, 0, 1);
	float dst_to_exit = dst_to_radius(pos, ray_dir, ground_radius + cloud_max_height);
	float march_length = dst_to_exit - dst_to_entry;

	vec3 dir_to_light = SUNDIR;
	vec3 atmosphere_sun = atmosphere(SUNDIR) * LIGHT0_ENERGY;
	vec3 atmosphere_ambient = atmosphere(normalize(vec3(1.0, 1.0, 0.0)));
	vec3 atmosphere_ground = atmosphere(normalize(vec3(1.0, -1.0, 0.0)));
	
	float cos_theta = dot(dir_to_light, ray_dir);
	float phase = dual_lobe_hg(cos_theta, eccentricity_1, eccentricity_2);

	float step_weight = 1. - ray_dir.y;
	float big_steps = mix(float(min_big_steps), float(max_big_steps), step_weight);
	float big_step_size = march_length / big_steps;
	float small_step_size = small_step_ratio * big_step_size;

	float offset_noise = 2. * texture(blue_noise, 2. * screen_uv).r - 1.; // [-1, 1];

	// Use hash function?
	float march_offset = march_offset_scale * offset_noise;

	vec3 march_start = pos + (dst_to_entry + 10. * hash(pos) * big_step_size) * ray_dir;

	float dst_marched = 0.;
	vec3 transmittance = vec3(1.);
	vec3 sun_energy = LIGHT0_COLOR * LIGHT0_ENERGY * SUN_ENERGY;
	vec3 light_energy = vec3(0.);

	bool is_small = false;
	int zero_density_steps = 0;

	float alpha = 1.;

	while(dst_marched < march_length) {
		vec3 march_pos = march_start + dst_marched * ray_dir;
		float conc = concentration(march_pos, is_small, 0.);

		float step_size = is_small ? small_step_size : big_step_size;
		//step_size = big_step_size;

		if (conc > 0.) {

			if (!is_small) {
				dst_marched -= big_step_size;
				is_small = true;
				zero_density_steps = 0;
				continue;
			}

			vec3 extinct = extinction(conc);
			vec3 absorb = absorbtion(conc);
			vec3 scatter = extinct - absorb;
			vec3 albedo = scatter / extinct;

			vec3 light_transmittance = sun_transmittance(march_pos);
			float rel_height = relative_height(march_pos);
			vec3 ambient = mix(atmosphere_ground, vec3(1), smoothstep(0, 1, rel_height))
				* mix(atmosphere_ambient, vec3(1), 0.4) * dir_to_light.y * density_multiplier * DENSITY_BASE_SCALE;

			vec3 integ = (1. - exp(-extinct * step_size));
			vec3 powder_int = integ-pow(transmittance, vec3(powder_strength))/(powder_strength + 1.) * (1. - exp(-(powder_strength + 1.) * extinct * step_size));

			light_energy += (ambient * integ + light_transmittance * atmosphere_sun * phase * powder_int) * albedo * transmittance;

			float a, b, c = 1.;
				
			for (int k = 1; k < scattering_octaves; k++) {
				a *= scattering_attenuation;
				b *= extinction_attenuation;
				c *= eccentricity_attenuation;
				
				
				light_transmittance = pow(light_transmittance, vec3(extinction_attenuation));
				albedo *= scattering_attenuation;
				phase = dual_lobe_hg(cos_theta, c * eccentricity_1, c * eccentricity_2);
				light_energy += (light_transmittance * atmosphere_sun * phase * powder_int) * albedo * transmittance;
			}
			
			transmittance *= exp(-extinct * step_size);
			alpha = min(min(transmittance.x, transmittance.y), transmittance.z);
			if (alpha < EPSILON) break;

		} else {
			if (is_small) {
				zero_density_steps++;
				if (zero_density_steps > int(floor(1./small_step_ratio))) {
					is_small = false;
				}
			}
		}
		dst_marched += step_size;
	}
	return vec4(light_energy, alpha);
}


void sky() {
	vec3 pos = POSITION + vec3(0, ground_radius, 0);
	vec3 ray_dir = normalize(EYEDIR);
	vec2 screen_uv = SCREEN_UV;

	vec3 light_pos = SUNDIR;
	float sun_cos_angle = dot(ray_dir, light_pos);

	if (AT_HALF_RES_PASS) {
		if (ray_dir.y > 0.) {
			vec4 cloud_data = cloud_march(pos, ray_dir, screen_uv);
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
		COLOR = transmittance * sky_col_base + cloud_color;
		COLOR = mix(COLOR, sky_col_base, smoothstep(0.6, 1.0, 1.0-ray_dir.y));

	}
}
