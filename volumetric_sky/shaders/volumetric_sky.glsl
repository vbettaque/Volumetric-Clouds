#[compute]
#version 450


layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Output texture (why 16 bit floats?).
layout(rgba16f, set = 0, binding = 0)
    uniform restrict writeonly image2D output_texture;

// Cloud and weather (noise) textures.
layout(set = 1, binding = 0) uniform sampler3D base_noise_texture;
layout(set = 1, binding = 1) uniform sampler3D detail_noise_texture;
layout(set = 1, binding = 2) uniform sampler2D weather_texture;

// Sky LUT (to be used later).
// layout(set = 2, binding = 0) uniform sampler2D sky_lut;

// Push constants (max size of 128 bytes (32 floats)).
// Data has to be aligned in blocks of 4 floats (vec4).
layout(push_constant, std430) uniform SkyUniforms {
	vec2 texture_size;
	vec2 update_position;

	vec2 wind_direction; 
	float wind_speed; // TODO: Unnecessary, combine with wind direction.
	float extinction_scale;

	vec4 ground_color;

	vec3 sun_direction; // Direction to sun.
	float sun_energy;

	vec3 sun_color;
	float time;

	vec2 pading;
	float cloud_coverage;
	float time_offset; // TODO: Unnecessary. Replace with weather offset?
} uniforms;


const vec3 RANDOM_VECTORS[6] = {
	vec3( 0.38051305f,  0.92453449f, -0.02111345f),
	vec3(-0.50625799f, -0.03590792f, -0.86163418f),
	vec3(-0.32509218f, -0.94557439f,  0.01428793f),
	vec3( 0.09026238f, -0.27376545f,  0.95755165f),
	vec3( 0.28128598f,  0.42443639f, -0.86065785f),
	vec3(-0.16852403f,  0.14748697f,  0.97460106f)
};

const float EPSILON = 0.01;
const float PI = 3.1415926535897;
const float FOUR_PI_INV = 0.0795774715459;

const float GROUND_RADIUS = 6371000.;
const float CLOUD_MIN_HEIGHT = 1500.;
const float CLOUD_MIN_RADIUS = GROUND_RADIUS + CLOUD_MIN_HEIGHT;
const float CLOUD_MAX_HEIGHT = 4000.;
const float CLOUD_MAX_RADIUS = GROUND_RADIUS + CLOUD_MAX_HEIGHT;

// Cloud gradients represented as vec4s.
// Smoothstep from 0 to 1 between v.x and and v.y.
// Smootstep from 1 to 0 between v.z and v.w.
const vec4 STRATUS_GRADIENT = vec4(0.02, 0.05, 0.09, 0.11);
const vec4 CUMULUS_GRADIENT = vec4(0.02, 0.2, 0.48, 0.625);
const vec4 CUMULONIMBUS_GRADIENT = vec4(0.01, 0.0625, 0.78, 1.);


// Scale for the weather texture.
// TODO: Eventually separate into base scale and adjustable scale?
const float WEATHER_UV_SCALE = 0.00006;

// Base scale for weather speed relative to wind speed.
// TODO: Eventually separate into base scale and adjustable scale?
const float WEATHER_WIND_SCALE = 16.;

// TODO: Eventually separate into base scale and adjustable scale?
const float BASE_UVW_SCALE = 0.00008;
const float BASE_WIND_SCALE = 12.; // Clouds base moves along with wind.
const vec3 BASE_FBM_WEIGHTS = vec3(0.625, 0.25, 0.125);

// TODO: Eventually separate into base scale and adjustable scale?
const float DETAIL_UVW_SCALE = 0.001;
const float DETAIL_SPEED = 40.; // Cloud detail speed relative to base.
const vec3 DETAIL_FBM_WEIGHTS = vec3(0.625, 0.25, 0.125);

const vec3 ALBEDO = vec3(1.);

// Dual-Lobe Henyey-Greenstein parameters;
const float PHASE_ECCENTRICITY_1 = -0.4; // [-1, 1]
const float PHASE_ECCENTRICITY_2 = 0.7; // [-1, 1]
const float PHASE_MIX_WEIGHT = 0.5; // [0, 1]


const float POWDER_COEFFICIENT = 2.;


// Light multi-scattering parameters:
const int SCATTERING_OCTAVES = 8;
const float EXTINCTION_ATTENUATION = 0.5;
const float SCATTERING_ATTENUATION = 0.5;
const float ECCENTRICITY_ATTENUATION = 0.5;

const int MIN_STEPS = 64;
const int MAX_STEPS = 128;
const int SUBSTEPS = 0; // [EPSILON, 1]


// Position hashing (https://www.shadertoy.com/view/4sfGzS).
// TODO: Replace with something else?
float hash(vec3 p) {
	p = fract(p * 0.3183099 + 0.1);
	p *= 17.;
	return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}


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
	float hg_1 = henyey_greenstein(cos_theta, g_scale * PHASE_ECCENTRICITY_1);
	float hg_2 = henyey_greenstein(cos_theta, g_scale * PHASE_ECCENTRICITY_2);
	return mix(hg_1, hg_2, PHASE_MIX_WEIGHT);
}


// Returns relative height within cloud layer.
// TODO: Account for curvature?
float relative_height(vec3 pos) {
    float rel_height = remap(pos.y, CLOUD_MIN_RADIUS, CLOUD_MAX_RADIUS, 0., 1.);
	return clamp(rel_height, 0., 1.);
}


// Returns distance to sphere with provided radius along the given direction.
// Assumes that pos is located within the sphere.
float dist_to_radius(vec3 pos, vec3 dir, float radius) {
	float a = dot(dir, dir);
    float b = 2. * dot(dir, pos);
    float c = dot(pos, pos) - (radius * radius);
	float d = sqrt((b * b) - 4. * a * c);
	float p = -b - d;
	float p2 = -b + d;
    return max(p, p2) / (2. * a);
}


// Linearly interpolates between the three different cloud gradients depending
// on the provided cloud type (between 0 and 1).
float cloud_gradient(float rel_height, float cloud_type) {
	float stratus = 1. - clamp(cloud_type * 2., 0., 1.);
	float cumulus = 1. - abs(cloud_type - 0.5) * 2.;
	float cumulonimbus = clamp(cloud_type - 0.5, 0., 1.) * 2.;
	vec4 gradient = stratus * STRATUS_GRADIENT
        + cumulus * CUMULUS_GRADIENT
        + cumulonimbus * CUMULONIMBUS_GRADIENT;
	return smoothstep(gradient.x, gradient.y, rel_height)
        - smoothstep(gradient.z, gradient.w, rel_height);
}


vec2 wind_offset() {
    return uniforms.time * uniforms.wind_speed 
        * normalize(uniforms.wind_direction);
}


// Samples from the weather texture.
// TODO: Change time offset to weather offset since it's not used elsewhere?
vec4 weather(vec2 pos) {
    vec2 weather_offset = WEATHER_WIND_SCALE * wind_offset();
	vec2 uv = WEATHER_UV_SCALE * (pos + weather_offset) + 0.5;
	return texture(weather_texture, uv);
}


// Analytic sky using Mie theory. 
// TODO: Implement it!
vec3 atmosphere() {
    return vec3(0., 0., 0.);
}


// Samples the cloud density (between 0 and 1).
float cloud_density(vec3 pos, bool has_detail, float lod) {
    float rel_height = relative_height(pos);
    vec4 weather = weather(pos.xz);
	float coverage = uniforms.cloud_coverage * weather.b;
	float gradient = cloud_gradient(rel_height, weather.r);

	if (gradient < EPSILON || coverage < EPSILON) return 0.;

    vec3 base_offset = vec3(0.);
    base_offset.xz = BASE_WIND_SCALE * wind_offset();

	vec3 base_uvw = BASE_UVW_SCALE * (pos + base_offset);
	vec4 base_noise = textureLod(base_noise_texture, base_uvw, lod - 2.);
	float base_fbm = dot(base_noise.gba, BASE_FBM_WEIGHTS);
	float base_density = remap(base_noise.r, -(1. - base_fbm), 1., 0., 1.); // TODO: Rethink validity
	base_density = remap(base_density * gradient, 1. - coverage, 1., 0., 1.);
	base_density *= coverage; // * clamp(4.*rel_height, 0, 1) ? 

	if (base_density < EPSILON) return 0.;
	if (!has_detail) return base_density;

    vec3 detail_offset = vec3(0., 1., 0.);
    detail_offset.xz = normalize(uniforms.wind_direction);
    detail_offset = base_density
        - normalize(detail_offset) * uniforms.time * DETAIL_SPEED;
	vec3 detail_uvw = DETAIL_UVW_SCALE * (pos + detail_offset);
	vec3 detail_noise = textureLod(detail_noise_texture, detail_uvw, lod).gba;
	float detail_fbm = dot(detail_noise, DETAIL_FBM_WEIGHTS);
	
	// Makes details "fluffy" at the bottom and "wispy" at the top.
	detail_fbm = mix(1. - detail_fbm, detail_fbm,
		clamp(10. * rel_height, 0., 1.)); // TODO: Identify scale paramter
	
	// Subtracts detail from base edges and renormalizes density to [0, 1].
	float density = remap(base_density,
		0.2 * detail_fbm * (1. - base_density) * rel_height, 1., 0., 1.); // TODO: Identify scale paramter

	// Reduces density at the bottom and increases density at the top.
	density = pow(clamp(density, 0, 1), (1. - rel_height) * 0.8 + 0.5); // TODO: Rewrite as lerp and identify relevant scales. Apply for base shape already?
	return density;
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
	float init_transmit_powder = pow(init_transmit, POWDER_COEFFICIENT + 1.);
	float extinct_powder = (POWDER_COEFFICIENT + 1.) * extinct;
	float powder = int_beer(init_transmit_powder, extinct_powder, step_size)
		/ (POWDER_COEFFICIENT + 1.);
	return beer - powder;
}


// TODO: Banding at low sun angles
float sun_transmittance(vec3 pos) {
    const int STEPS = 6;
    const float STEPS_SQ = float(STEPS * STEPS);
    const float STEP_SIZE = (CLOUD_MAX_HEIGHT - CLOUD_MIN_HEIGHT) / STEPS_SQ;
	float total_density = 0.;
	vec3 sample_pos = pos;
	for (int i = 0; i < STEPS; i++) {
		sample_pos += STEP_SIZE
            * (uniforms.sun_direction + float(i) * RANDOM_VECTORS[i]);
		total_density += cloud_density(sample_pos, true, float(i));
	}
	sample_pos = pos + STEPS_SQ * STEP_SIZE * uniforms.sun_direction / 2.;
	float rel_height = relative_height(sample_pos);
	total_density += cloud_density(sample_pos, true, 5.);
	float extinction = uniforms.extinction_scale * total_density;
	return beer(extinction, STEP_SIZE);
}


vec3 scatter_radiance(vec3 pos, float cos_theta) {
	vec3 radiance = uniforms.sun_energy * uniforms.sun_color; // TODO: Add sky LUT value
	float transmittance = sun_transmittance(pos);
	float phase = dual_lobe_hg(cos_theta, 1.);
	float multi_scatter = transmittance * phase;
	float a = 1; float b = 1.;
	for (int k = 1; k < SCATTERING_OCTAVES; k++) {
		transmittance = pow(transmittance, EXTINCTION_ATTENUATION);
		a *= SCATTERING_ATTENUATION;
		b *= ECCENTRICITY_ATTENUATION;

		phase = dual_lobe_hg(cos_theta, b);
		multi_scatter += a * transmittance * phase;
	}
	return multi_scatter * radiance;
}


vec3 ambient_radiance(vec3 pos) {
	return vec3(0.);
}


bool detail_march(
	vec3 march_start, vec3 march_step, int steps, inout vec3 radiance,
	inout float transmit
) {
	float step_size = length(march_step);
	vec3 march_dir = march_step / step_size;
	float cos_theta = dot(uniforms.sun_direction, march_dir);

	vec3 march_pos = march_start - march_step;
	bool has_density = false;

	for(int i = 0; i < steps; i++) {
		march_pos += march_step;
		float density = cloud_density(march_pos, true, 0.);

		if (density <= 0.) continue;
		has_density = true;

		float extinction = uniforms.extinction_scale * density;

		vec3 ambient = ambient_radiance(march_pos);
		vec3 scatter = scatter_radiance(march_pos, cos_theta);
		float beer_powder = int_beer_powder(transmit, extinction, step_size);

		radiance += beer_powder * (ambient + scatter) * ALBEDO;
		transmit *= beer(extinction, step_size);
		if (transmit < EPSILON) break;
	}
	return has_density;
}


vec4 cloud_march(vec3 march_start, vec3 march_step, int steps) {
	vec3 radiance = vec3(0.);
	float transmittance = 1.;

	if (SUBSTEPS <= 0) {
		detail_march(march_start, march_step, steps, radiance, transmittance);
		return vec4(radiance, 1. - transmittance);
	}

	float step_size = length(march_step);
	vec3 march_dir = march_step / step_size;
	vec3 march_pos = march_start - march_step;

	float substep_size = step_size / SUBSTEPS;
	vec3 march_substep = substep_size * march_dir;

	bool using_substeps = false;

	for(int i = 0; i < steps; i++) {
		march_pos += march_step;
		float density = cloud_density(march_pos, false, 0.);

		if (density > 0. && !using_substeps) {
			detail_march(
				march_pos - march_step, march_substep, SUBSTEPS,
				radiance, transmittance
			);
			using_substeps = true;
		} 

		if (using_substeps){
			using_substeps = detail_march(
				march_pos, march_substep, SUBSTEPS, radiance, transmittance
			);
		}
	}
	return vec4(radiance, 1. - transmittance);
}

// TODO: Add random offset to start postion.
vec4 sky_lut(vec3 dir) {
	if (dir.y <= 0.) return vec4(0.); // Only draw clouds above horizon.

	vec3 pos = vec3(0., GROUND_RADIUS, 0.); // TODO: Adjust for player position (and planet curvature)?

	float dist_to_entry = dist_to_radius(pos, dir, CLOUD_MIN_RADIUS);
	float dist_to_exit = dist_to_radius(pos, dir, CLOUD_MAX_RADIUS);
	float march_dist = dist_to_exit - dist_to_entry;

	// Decrease number of big march steps towards the horizon.
	float steps = mix(MIN_STEPS, MAX_STEPS, dir.y);

	// float offset_noise = 2. * texture(blue_noise, 2. * screen_uv).r - 1.; // [-1, 1];
	// float march_offset = march_offset_scale * offset_noise;
	// vec3 march_start = pos + (dst_to_entry + 10. * hash(pos) * big_step_size) * ray_dir;

	vec3 march_step = dir * march_dist / steps;
	vec3 march_start = pos + dist_to_entry * dir + march_step / 2.; // TODO: Add noise to offset

	return cloud_march(march_start, march_step, int(steps));
}


void main() {

}
