#[compute]
#version 450


layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// Output texture (why 16 bit floats?).
layout(rgba16f, set = 0, binding = 0)
    uniform restrict writeonly image2D output_texture;

// Cloud and weather (noise) textures.
layout(set = 1, binding = 0) uniform sampler3D base_noise;
layout(set = 1, binding = 1) uniform sampler3D detail_noise;
layout(set = 1, binding = 2) uniform sampler2D weather_texture;

// Sky LUT (to be used later).
// layout(set = 2, binding = 0) uniform sampler2D sky_lut;

// Push constants (max size of 128 bytes (32 floats)).
// Data has to be aligned in blocks of 4 floats (vec4).
layout(push_constant, std430) uniform SkyUniforms {
	vec2 texture_size;
	vec2 update_position;

	vec2 wind_direction; // TODO: Combine with wind speed.
	float wind_speed;
	float extinction_scale;

	vec4 ground_color;

	vec3 sun_direction;
	float sun_energy;

	vec3 sun_color;
	float time;

	vec2 pading;
	float cloud_coverage;
	float time_offset; // TODO: Replace with weather offset?
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
const float CLOUD_MAX_HEIGHT = 4000.;

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
const float BASE_WIND_SCALE = 12.; // Clouds base moves relative to wind.
const vec3 BASE_FBM_WEIGHTS = vec3(0.625, 0.25, 0.125);

// TODO: Eventually separate into base scale and adjustable scale?
const float DETAIL_UVW_SCALE = 0.001;
const float DETAIL_SPEED = 40.; // Cloud details moves with own speed relative to base.
const vec3 DETAIL_FBM_WEIGHTS = vec3(0.625, 0.25, 0.125);


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
float dual_lobe_hg(float cos_theta, float g_1, float g_2, float mix_weight) {
	float hg_1 = henyey_greenstein(cos_theta, g_1);
	float hg_2 = henyey_greenstein(cos_theta, g_2);
	return mix(hg_1, hg_2, mix_weight);
}


// Returns relative height within cloud layer.
// TODO: Account for curvature?
float relative_height(vec3 pos) {
	const float CLOUD_MIN_RADIUS = GROUND_RADIUS + CLOUD_MIN_HEIGHT;
    const float CLOUD_MAX_RADIUS = GROUND_RADIUS + CLOUD_MAX_HEIGHT;
    float rel_height = remap(pos.y, CLOUD_MIN_RADIUS, CLOUD_MAX_RADIUS, 0., 1.);
	return clamp(rel_height, 0., 1.);
}


// Returns distance to sphere with provided radius along the given direction.
// Assumes that pos is located within the sphere.
float dst_to_radius(vec3 pos, vec3 dir, float radius) {
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
float density(vec3 pos, bool has_detail, float lod) {
    float rel_height = relative_height(pos);
    vec4 weather = weather(pos.xz);
	float coverage = uniforms.cloud_coverage * weather.b;
	float gradient = cloud_gradient(rel_height, weather.r);

	if (gradient < EPSILON || coverage < EPSILON) return 0.;

    vec3 base_offset = vec3(0.);
    base_offset.xz = BASE_WIND_SCALE * wind_offset();

	vec3 base_uvw = BASE_UVW_SCALE * (pos + base_offset);
	vec4 base_noise = textureLod(base_noise, base_uvw, lod - 2.);
	float base_fbm = dot(base_noise.gba, BASE_FBM_WEIGHTS);
	float base_density = remap(base_noise.r, -(1. - base_fbm), 1., 0., 1.);
	base_density = remap(base_density * gradient, 1. - coverage, 1., 0., 1.);
	base_density *= coverage;

	if (base_density < EPSILON) return 0.;
	if (!has_detail) return base_density;

    vec3 detail_offset = vec3(0., 1., 0.);
    detail_offset.xz = normalize(uniforms.wind_direction);
    detail_offset = base_density
        - normalize(detail_offset) * uniforms.time * DETAIL_SPEED;

	vec3 detail_uvw = DETAIL_UVW_SCALE * (pos + detail_offset);
	vec3 detail_noise = textureLod(detail_noise, detail_uvw, lod).gba;
	float detail_fbm = dot(detail_noise, DETAIL_FBM_WEIGHTS);
	// What's going on here?!
	detail_fbm = mix(detail_fbm, 1. - detail_fbm, clamp(4. * rel_height, 0., 1.));
	float density = remap(base_density, 0.4 * detail_fbm * rel_height, 1., 0., 1.);
	density = pow(clamp(density, 0, 1), (1. - rel_height) * 0.8 + 0.5);
	return density;
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
		total_density += density(sample_pos, true, float(i));
	}
	sample_pos = pos + STEPS_SQ * STEP_SIZE * uniforms.sun_direction / 2.;
	float rel_height = relative_height(sample_pos);
	total_density += density(sample_pos, true, 5.);
	float extinction = uniforms.extinction_scale * total_density;
	return exp(-extinction * STEP_SIZE);
}


void main() {

}
