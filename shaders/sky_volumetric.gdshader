shader_type sky;
render_mode use_half_res_pass;

const float EPSILON = 0.01;

uniform ivec2 viewport_size;

group_uniforms Sky;
uniform vec3 base_color_a: source_color = vec3(0.78, 0.86, 1.);
uniform vec3 base_color_b: source_color = vec3(0.4, 0.6, 0.9);

group_uniforms Clouds;
uniform float base_radius = 150000;
uniform float cloud_base_height = 1500.;
uniform float cloud_max_height = 4000.;

group_uniforms CloudMarch;
uniform int min_steps: hint_range(1, 200, 1) = 64;
uniform int max_steps: hint_range(1, 200, 1) = 128;
uniform float march_offset_scale = 1.;

group_uniforms LightMarch;
uniform int light_march_steps: hint_range(1, 100, 1) = 4;
uniform float light_march_absorption = 1.;

group_uniforms Noise;
uniform sampler2D blue_noise: repeat_enable;
uniform sampler3D perlin_worley: repeat_enable;


float dst_to_height(vec3 ray_dir, float height) {
	float p = base_radius * ray_dir.y;
	float q = (2. * base_radius + height) * height;
	return -p + sqrt(p * p + q);
}

float sample_density(vec3 pos) {
	return (clamp(texture(perlin_worley, pos / 10000.).r, 0.5, 1.) - 0.5) / 500.;
}

float light_march(vec3 pos) {
	vec3 dir_to_light = normalize(LIGHT0_DIRECTION);

	// Make more exact?
	float march_length = (cloud_max_height - cloud_base_height) / 2.;
	float step_size = march_length / float(light_march_steps);

	vec3 march_start = pos + step_size/2. * dir_to_light;

	float density = 0.;
	for (int k = 0; k < light_march_steps; k++) {
		vec3 march_pos = march_start + float(k) * step_size * dir_to_light;
		density += sample_density(march_pos) * step_size;
	}
	return exp(-light_march_absorption * density);
}

vec4 cloud_march(vec3 pos, vec3 ray_dir, vec2 screen_uv) {
	float dst_to_entry = dst_to_height(ray_dir, cloud_base_height);
	float dst_to_exit = dst_to_height(ray_dir, cloud_max_height);
	float march_length = dst_to_exit - dst_to_entry;
	
	float step_weight = 1. - ray_dir.y;
	float steps = mix(float(min_steps), float(max_steps), step_weight);
	float step_size = march_length / steps;

	float offset_noise = 2. * texture(blue_noise, 2. * screen_uv).r - 1.; // [-1, 1];
	float march_offset = march_offset_scale * offset_noise;
	
	vec3 march_start = pos + (dst_to_entry + march_offset) * ray_dir;

	float dst_marched = 0.;
	float transmittance = 1.;
	vec3 light_energy = vec3(0.);
	
	while(dst_marched < march_length) {
		vec3 march_pos = march_start + dst_marched * ray_dir;
		float density = sample_density(march_pos);

		if (density > 0.) {
			
			transmittance *= exp(-density * step_size);

//			float light_transmittance = light_march(march_pos);
//			light_energy += transmittance * light_transmittance * step_size;
			
			if (transmittance < EPSILON) break;
		}
		dst_marched += step_size;
	}
	return vec4(light_energy, transmittance);
}

void sky() {
	vec3 pos = POSITION;
	vec3 ray_dir = normalize(EYEDIR);
	vec2 screen_uv = SCREEN_UV;

	vec3 light_pos = LIGHT0_DIRECTION;
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
		vec3 light_energy = LIGHT0_ENERGY * HALF_RES_COLOR.rgb;
		
		float sky_col_weight = sqrt(max(ray_dir.y, 0));
		vec3 sky_col_base = mix(base_color_a, base_color_b, sky_col_weight);
	
		vec3 cloud_color = light_energy * LIGHT0_COLOR;
		COLOR = sky_col_base * transmittance + vec3(1.) * (1. - transmittance);
	}
}
