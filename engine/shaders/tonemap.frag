#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location=0) in vec2 fragUV;

layout(location=0) out vec4 outColor;

uint hash(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

// Fixed per pixel (like the photo's own grain), standard normal
float gaussianNoise(uvec2 pixel, uint channel) {
	uint seed = hash(pixel.x ^ hash(pixel.y ^ hash(channel + 0x51ed27u)));
	float u1 = max(float(hash(seed) >> 8) / 16777216.0, 1e-7);
	float u2 = float(hash(seed ^ 0x9e3779b9u) >> 8) / 16777216.0;
	return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

// The photo at a framebuffer pixel; false outside the 3D view
bool photoAt(ivec2 pixel, out vec3 photo) {
	vec2 uv = (vec2(pixel) + 0.5 - frame.viewport.xy) / frame.viewport.zw;
	photo = textureLod(textures[uint(frame.whiteBalance.w)], uv, 0.0).rgb;
	return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)));
}

void main() {
	uint hdrTexture = materials[push.materialIndex].baseColorTexture;
	ivec2 size = textureSize(textures[hdrTexture], 0);
	ivec2 pixel = ivec2(gl_FragCoord.xy);
	vec4 center = texelFetch(textures[hdrTexture], pixel, 0);
	vec3 hdr = center.rgb;

	// Camera matching: the photo already went through its camera, so only what the object changed (itself and its
	// shadows) gets the camera's blur. Grain goes on the object only (alpha = how much of the pixel it covers)
	vec3 photo;
	if (frame.exposure.w > 0.5 && photoAt(pixel, photo)) {
		float blur = frame.exposure.z;
		int radius = min(int(ceil(blur * 2.5)), 8);
		if (radius > 0) {
			vec3 change = vec3(0.0);
			float weightSum = 0.0;
			for (int y = -radius; y <= radius; y++) {
				for (int x = -radius; x <= radius; x++) {
					ivec2 at = clamp(pixel + ivec2(x, y), ivec2(0), size - 1);
					vec3 neighbourPhoto;
					if (!photoAt(at, neighbourPhoto)) continue;
					float weight = exp(-float(x * x + y * y) / (2.0 * blur * blur));
					change += (texelFetch(textures[hdrTexture], at, 0).rgb - neighbourPhoto) * weight;
					weightSum += weight;
				}
			}
			hdr = photo + change / max(weightSum, 1e-6);
		}

		float grain = frame.exposure.y * center.a;
		if (grain > 0.0) {
			vec3 noise = vec3(gaussianNoise(uvec2(pixel), 0u), gaussianNoise(uvec2(pixel), 1u), gaussianNoise(uvec2(pixel), 2u));
			hdr = pow(max(sqrt(max(hdr, vec3(0.0))) + grain * noise, vec3(0.0)), vec3(2.0));
		}
	}

	vec3 color = hdr * frame.exposure.x * frame.whiteBalance.rgb;
	outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
