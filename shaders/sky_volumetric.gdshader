shader_type sky;

const float EPSILON = 0.01;

uniform ivec2 viewport_size;

group_uniforms Sky;
uniform vec3 base_color_a: source_color = vec3(0.78, 0.86, 1.);
uniform vec3 base_color_b: source_color = vec3(0.4, 0.6, 0.9);

group_uniforms Clouds;
uniform float cloud_base_height = 400.;
uniform float cloud_max_height = 1000.;

group_uniforms CloudMarch;
uniform int cloud_march_steps: hint_range(1, 100, 1) = 10;
uniform float cloud_march_offset_scale = 1.;

group_uniforms LightMarch;
uniform int light_march_steps: hint_range(1, 100, 1) = 4;
uniform float light_march_absorption = 1.;

group_uniforms Noise;
uniform sampler2D blue_noise: repeat_enable;
uniform sampler3D perlin_worley: repeat_enable;

float sample_density(vec3 pos) {
	return 0.01;
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
	float dst_to_entry = cloud_base_height / abs(ray_dir.y);
	vec3 entry_point = pos + dst_to_entry * ray_dir;
	
	float cloud_thickness = cloud_max_height - cloud_base_height;
	float march_length = cloud_thickness / abs(ray_dir.y);
	
	float step_size = march_length / float(cloud_march_steps);

	float offset_noise = 2. * texture(blue_noise, 2. * screen_uv).r - 1.; // [-1, 1];
	float march_offset = cloud_march_offset_scale * offset_noise;
	
	vec3 march_start = entry_point + march_offset * ray_dir;
	
	float dst_marched = 0.;
	float transmittance = 1.;
	vec3 light_energy = vec3(0.);
	
	while(dst_marched < march_length) {
		vec3 march_pos = march_start + dst_marched * ray_dir;
		float density = sample_density(march_pos);

		if (density > 0.) {
			transmittance *= exp(-density * step_size);

			float light_transmittance = light_march(march_pos);
			light_energy += transmittance * light_transmittance * step_size;
			
			if (transmittance < EPSILON) break;
		}
		dst_marched += step_size;
	}
	
	return vec4(transmittance, light_energy);
}

void sky() {
	vec3 pos = POSITION;
	vec3 ray_dir = normalize(EYEDIR);
	vec2 screen_uv = SCREEN_UV;

	vec3 light0_pos = LIGHT0_DIRECTION;
	float sun_cos_angle = dot(ray_dir, light0_pos);
	
	float transmittance = 1.;
	vec3 light_energy = vec3(0.);
	
	if (ray_dir.y > 0.) {
		vec4 cloud_data = cloud_march(pos, ray_dir, screen_uv);
		light_energy = LIGHT0_ENERGY * cloud_data.rgb;
		transmittance = cloud_data.a;
	}
	
	float sky_col_weight = sqrt(max(ray_dir.y, 0));
	vec3 sky_col_base = mix(base_color_a, base_color_b, sky_col_weight);
	
	vec3 cloud_color = light_energy * LIGHT0_COLOR;
	COLOR = sky_col_base * transmittance + cloud_color;
}
