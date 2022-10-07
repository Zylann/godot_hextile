shader_type spatial;

// Ported from https://github.com/mmikk/hextile-demo

uniform sampler2D u_albedo_texture : hint_albedo;
uniform sampler2D u_normal_texture;
uniform float u_show_weights;
uniform float u_tile_rate;
uniform float u_tile_rot_strength;
uniform float u_use_regular_tiling;
uniform float u_roughness = 0.5;

float fmod(float x, float y) {
	return (x - y * trunc(x / y));
}

float module_oper(float a, float b) {
	return a - (b * floor(a / b));
}

vec3 gain3(vec3 x, float r) {
	float k = log(1.0 - r) / log(0.5);
	vec3 s = 2.0 * step(0.5, x);
	vec3 m = 2.0 * (1.0 - s);
	vec3 res = 0.5 * s + 0.25 * m * pow(max(vec3(0.0), s + x * m), vec3(k));
	return res.xyz / (res.x + res.y + res.z);
}

const float PI = 3.14159;

mat2 load_rot_2x2(vec2 idx, float rot_strength) {
	float angle = abs(idx.x * idx.y) + abs(idx.x + idx.y) + PI;
	
	// Remap to +/-pi
	//angle = fmod(angle, 2.0*PI);
	if (angle < 0.0) {
		angle += 2.0 * PI;
	}
	if (angle > PI) {
		angle -= 2.0 * PI;
	}
	
	angle *= rot_strength;
	
	float cs = cos(angle);
	float si = sin(angle);
	
	//return mat2(vec2(cs, -si), vec2(si, cs));
	return mat2(vec2(cs, si), vec2(-si, cs));
}

vec2 make_cen_st(vec2 vertex) {
	//mat2 inv_skew_mat = mat2(vec2(1.0, 0.5), vec2(0.0, 1.0 / 1.15470054));
	mat2 inv_skew_mat = mat2(vec2(1.0, 0.0), vec2(0.5, 1.0 / 1.15470054));
	return (inv_skew_mat * vertex) / (2.0 * sqrt(3.0));
}

vec3 produce_hex_weights(vec3 w, vec2 vertex1, vec2 vertex2, vec2 vertex3) {
	vec3 res = vec3(0.0);
	float v1 = module_oper(((vertex1.x - vertex1.y)), 3.0);
	if (v1 < 0.0) {
		v1 += 3.0;
	}
	
	float vh = v1 < 2.0 ? (v1 + 1.0) : 0.0;
	float vl = v1 > 0.0 ? (v1 - 1.0) : 2.0;
	float v2 = vertex1.x < vertex3.x ? vl : vh;
	float v3 = vertex1.x < vertex3.x ? vh : vl;
	
	res.x = v3 == 0.0 ? w.z : (v2 == 0.0 ? w.y : w.x);
	res.y = v3 == 1.0 ? w.z : (v2 == 1.0 ? w.y : w.x);
	res.z = v3 == 2.0 ? w.z : (v2 == 2.0 ? w.y : w.x);
	
	return res;
}

vec2 hash(vec2 p) {
	vec2 r = mat2(vec2(127.1, 311.7), vec2(269.5, 183.3)) * p;
	return fract(sin(r) * 43758.5453);
}

// Given a point in UV, compute local triangle barycentric coordinates and vertex IDs
void triangle_grid(
	out float w1, out float w2, out float w3,
	out vec2 vertex1, out vec2 vertex2, out vec2 vertex3,
	vec2 uv) {
		
	// Scaling of the input
	// controls the size of the input with respect to the size of the tiles. 
	uv *= 2.0 * sqrt(3.0);
	
	// Skew input space into simplex triangle grid
	//mat2 grid_to_skewed_grid = mat2(vec2(1.0, -0.57735027), vec2(0.0, 1.15470054));
	mat2 grid_to_skewed_grid = mat2(vec2(1.0, 0.0), vec2(-0.57735027, 1.15470054));
	vec2 skewed_coords = grid_to_skewed_grid * uv;
	
	vec2 base_id = floor(skewed_coords);
	vec3 temp = vec3(fract(skewed_coords), 0.0);
	temp.z = 1.0 - temp.x - temp.y;
	
	float s = step(0.0, -temp.z);
	float s2 = 2.0 * s - 1.0;
	
	w1 = -temp.z * s2;
	w2 = s - temp.y * s2;
	w3 = s - temp.x * s2;
	
	vertex1 = base_id + vec2(s, s);
	vertex2 = base_id + vec2(s, 1.0 - s);
	vertex3 = base_id + vec2(1.0 - s, s);
}

void hex2col_tex(out vec4 color, out vec3 weights, vec2 uv, float rot_strength, float r) {
	// compute uv derivatives
	vec2 dstdx = dFdx(uv);
	vec2 dstdy = dFdy(uv);
	
	// Get triangle info
	float w1;
	float w2;
	float w3;
	vec2 vertex1;
	vec2 vertex2;
	vec2 vertex3;
	triangle_grid(w1, w2, w3, vertex1, vertex2, vertex3, uv);

	mat2 rot1 = load_rot_2x2(vertex1, rot_strength);
	mat2 rot2 = load_rot_2x2(vertex2, rot_strength);
	mat2 rot3 = load_rot_2x2(vertex3, rot_strength);

	vec2 cen1 = make_cen_st(vertex1);
	vec2 cen2 = make_cen_st(vertex2);
	vec2 cen3 = make_cen_st(vertex3);

	// assign random offset to each triangle vertex
	// this is used later to fetch from texture
	vec2 uv1 = (uv - cen1) * rot1 + cen1 + hash(vertex1);
	vec2 uv2 = (uv - cen2) * rot2 + cen2 + hash(vertex2);
	vec2 uv3 = (uv - cen3) * rot3 + cen3 + hash(vertex3);

	// Fetch input
	// We could simply use texture2D function, however, the sreen space derivatives could be broken 
	// since we are using random offsets, we use texture2DGrad to make sure that we pass correct derivatives explicitly.
	vec4 c1 = textureGrad(u_albedo_texture, uv1, dstdx * rot1, dstdy * rot1);
	vec4 c2 = textureGrad(u_albedo_texture, uv2, dstdx * rot2, dstdy * rot2);
	vec4 c3 = textureGrad(u_albedo_texture, uv3, dstdx * rot3, dstdy * rot3);

	// use luminance as weight
	vec3 lw = vec3(0.299, 0.587, 0.114);
	vec3 dw = vec3(dot(c1.xyz, lw), dot(c2.xyz, lw), dot(c3.xyz, lw));

	dw = mix(vec3(1.0), dw, 0.6); // 0.6 is fall off constant
	vec3 w = dw * pow(vec3(w1, w2, w3), vec3(7.0)); // 7 is g_exp
	w /= (w.x + w.y + w.z);
	if (r != 0.5) {
		w = gain3(w, r);
	}

	// blend weights with color linearly 
	// histogram preserving blending will be better but requires precompution step to create histogram texture
	color = w.x * c1 + w.y * c2 + w.z * c3;
	weights = produce_hex_weights(w.xyz, vertex1, vertex2, vertex3);
}

// Input: vm is tangent space normal in [-1;1].
// Output: convert vm to a derivative.
vec2 tspace_normal_to_derivative(vec3 vm) {
	float scale = 1.0 / 128.0;

	// Ensure vm delivers a positive third component using abs() and
	// constrain vm.z so the range of the derivative is [-128; 128].
	vec3 vma = abs(vm);
	float z_ma = max(vma.z, scale*max(vma.x, vma.y));

	// Set to match positive vertical texture coordinate axis.
	bool g_flip_vert_deriv = true;
	float s = g_flip_vert_deriv ? -1.0 : 1.0;
	return -vec2(vm.x, s * vm.y) / z_ma;
}

vec2 sample_deriv(sampler2D nmap, vec2 st, vec2 dstdx, vec2 dstdy) {
	vec3 vm = 2.0 * textureGrad(nmap, st, dstdx, dstdy).rgb - vec3(1.0);
	return tspace_normal_to_derivative(vm);
}

void bump_hex2deriv_nmap(out vec2 deriv, out vec3 weights, sampler2D nmap, vec2 st,
	float rot_strength, float r) {
	
	vec2 dstdx = dFdx(st);
	vec2 dstdy = dFdy(st);

	// Get triangle info
	float w1;
	float w2;
	float w3;
	vec2 vertex1;
	vec2 vertex2;
	vec2 vertex3;
	triangle_grid(w1, w2, w3, vertex1, vertex2, vertex3, st);

	mat2 rot1 = load_rot_2x2(vertex1, rot_strength);
	mat2 rot2 = load_rot_2x2(vertex2, rot_strength);
	mat2 rot3 = load_rot_2x2(vertex3, rot_strength);

	vec2 cen1 = make_cen_st(vertex1);
	vec2 cen2 = make_cen_st(vertex2);
	vec2 cen3 = make_cen_st(vertex3);

	vec2 st1 = (st - cen1) * rot1 + cen1 + hash(vertex1);
	vec2 st2 = (st - cen2) * rot2 + cen2 + hash(vertex2);
	vec2 st3 = (st - cen3) * rot3 + cen3 + hash(vertex3);

	// Fetch input
	vec2 d1 = sample_deriv(nmap, st1, dstdx * rot1, dstdy * rot1);
	vec2 d2 = sample_deriv(nmap, st2, dstdx * rot2, dstdy * rot2);
	vec2 d3 = sample_deriv(nmap, st3, dstdx * rot3, dstdy * rot3);

	d1 = rot1 * d1;
	d2 = rot2 * d2;
	d3 = rot3 * d3;

	// produce sine to the angle between the conceptual normal
	// in tangent space and the Z-axis
	vec3 d = vec3(dot(d1, d1), dot(d2, d2), dot(d3, d3));
	vec3 dw = sqrt(d / (1.0 + d));
	
	float g_fall_off_contrast = 0.6;
	float g_exp = 7.0;
	dw = mix(vec3(1.0), dw, g_fall_off_contrast);
	vec3 w = dw * pow(vec3(w1, w2, w3), vec3(g_exp));
	w /= (w.x + w.y + w.z);
	if (r != 0.5) {
		w = gain3(w, r);
	}

	deriv = w.x * d1 + w.y * d2 + w.z * d3;
	weights = produce_hex_weights(w.xyz, vertex1, vertex2, vertex3);
}

// Used to produce a surface gradient from the gradient of a volume
// bump function such as 3D Perlin noise. Equation 2 in [Mik10].
vec3 surfgrad_from_volume_gradient(vec3 grad, vec3 base_normal) {
	return grad - dot(base_normal, grad) * base_normal;
}

vec3 resolve_normal_from_surface_gradient(vec3 surf_grad, vec3 base_normal) {
	return normalize(base_normal - surf_grad);
}

float get_tile_rate() {
	return 0.05 * u_tile_rate;
}

void fetch_color_and_weight(out vec3 color, out vec3 weights, vec2 uv) {
	vec4 col4;
	hex2col_tex(col4, weights, uv, u_tile_rot_strength, 0.7);
	color = col4.xyz;
}

void fetch_deriv_and_weight(out vec2 dhduv, out vec3 weights, vec2 st) {
	float g_fake_contrast_normal = 0.7;
	bump_hex2deriv_nmap(dhduv, weights, u_normal_texture, st, u_tile_rot_strength, g_fake_contrast_normal);
}

vec3 determine_triplanar_weights(float k, vec3 base_normal) {
	vec3 weights = abs(base_normal) - 0.2;
	weights = max(vec3(0.0), weights);
	weights = pow(weights, vec3(k));
	weights /= (weights.x + weights.y + weights.z);
	return weights;
}

varying vec3 v_position_world;
varying vec3 v_normal_world;

void vertex() {
	v_position_world = VERTEX;
	v_normal_world = (WORLD_MATRIX * vec4(NORMAL, 0.0)).xyz;
}

void fragment() {
	vec3 surf_pos_in_world = v_position_world;
	
	vec3 sp = get_tile_rate() * surf_pos_in_world;
	vec2 uv0 = vec2(sp.x, sp.z);
	
	if (u_use_regular_tiling > 0.0) {
		ALBEDO = texture(u_albedo_texture, uv0).rgb;
		NORMALMAP = texture(u_normal_texture, uv0).rgb;

	} else {
		vec3 color;
		vec3 weights;
		//vec3 nm;
		fetch_color_and_weight(color, weights, uv0);
		
		vec2 dhduv;
		vec3 weights2;
		fetch_deriv_and_weight(dhduv, weights2, uv0);
		
//		dhduv.y *= -1.0;
//		dhduv.y *= -1.0;

		vec3 vol_grad = vec3(dhduv.x, 0.0, dhduv.y);
		vec3 base_normal = v_normal_world;//vec3(0.0, 1.0, 0.0);
		vec3 surf_grad = surfgrad_from_volume_gradient(vol_grad, base_normal);
		// TODO I'm not sure what triplanar has to do here, but I just ported the code
		float weight_y = determine_triplanar_weights(1.0, base_normal).y;
		float bump_intensity = 1.0;
		surf_grad *= (weight_y * bump_intensity);
		vec3 normal = resolve_normal_from_surface_gradient(surf_grad, base_normal);
		
		normal = (INV_CAMERA_MATRIX * vec4(normal, 0.0)).xyz;
		
		if (u_show_weights > 0.0) {
			ALBEDO = weights;
			//NORMALMAP = vec3(0.5, 0.5, 1.0);
			
		} else {
			ALBEDO = color;
			NORMAL = normal;
			// Using NORMALMAP is inconvenient...
			//NORMALMAP = nm;
		}
	}
	ROUGHNESS = u_roughness;
}
