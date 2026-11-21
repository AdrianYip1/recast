// Point lights with cube shadow maps. Needs common.glsl first.

const float SHADOW_NORMAL_OFFSET = 0.02;
const float SHADOW_BIAS = 0.0005;

uint shadowFace(vec3 lightToPoint) {
	vec3 a = abs(lightToPoint);
	if (a.x >= a.y && a.x >= a.z) return lightToPoint.x > 0.0 ? 0u : 1u;
	if (a.y >= a.z) return lightToPoint.y > 0.0 ? 2u : 3u;
	return lightToPoint.z > 0.0 ? 4u : 5u;
}

float shadowVisibility(uint atlas, uint lightIndex, vec3 worldPos, vec3 normal) {
	Light light = frame.lights[lightIndex];
	if (light.position.w < 0.5) return 1.0;

	vec3 offsetPos = worldPos + normal * SHADOW_NORMAL_OFFSET;
	uint face = light.intensity.w > 0.5 ? 0u : shadowFace(offsetPos - light.position.xyz);
	vec4 clip = light.faceViewProj[face] * vec4(offsetPos, 1.0);
	vec3 ndc = clip.xyz / clip.w;

	float faceResolution = float(frame.lightInfo.w);
	vec2 atlasSize = vec2(6.0, float(MAX_LIGHTS)) * faceResolution;
	vec2 faceUV = clamp(ndc.xy * 0.5 + 0.5, vec2(3.0 / faceResolution), vec2(1.0 - 3.0 / faceResolution));
	vec2 texel = (vec2(float(face), float(lightIndex)) + faceUV) * faceResolution - 0.5;
	vec2 f = fract(texel);
	vec2 base = floor(texel) + 0.5;

	float visible = 0.0;
	for (int y = -1; y <= 2; y++) {
		float wy = y == -1 ? 1.0 - f.y : (y == 2 ? f.y : 1.0);
		for (int x = -1; x <= 2; x++) {
			float wx = x == -1 ? 1.0 - f.x : (x == 2 ? f.x : 1.0);
			float depth = texture(textures[atlas], (base + vec2(x, y)) / atlasSize).r;
			visible += (ndc.z - SHADOW_BIAS <= depth) ? wx * wy : 0.0;
		}
	}
	return visible / 9.0;
}

// Irradiance from the ambient term plus every point light.
// includeDynamic = false ignores dynamic casters: the lighting of the scene without the object
vec3 irradiance(vec3 worldPos, vec3 normal, bool includeDynamic) {
	uint lightCount = frame.lightInfo.x;
	if (lightCount == 0u) {
		return frame.ambient.rgb + vec3(0.65 * abs(dot(normal, -frame.lightDir.xyz)));
	}

	vec3 result = frame.ambient.rgb;
	for (uint i = 0u; i < lightCount; i++) {
		Light light = frame.lights[i];
		bool directional = light.intensity.w > 0.5;
		vec3 toLight = directional ? -light.direction.xyz : light.position.xyz - worldPos;
		float distanceSquared = directional ? 1.0 : max(dot(toLight, toLight), 1e-4);
		float cosine = max(dot(normal, toLight * inversesqrt(dot(toLight, toLight))), 0.0);
		if (cosine <= 0.0) continue;

		float visibility = shadowVisibility(frame.lightInfo.y, i, worldPos, normal);
		if (includeDynamic) visibility *= shadowVisibility(frame.lightInfo.z, i, worldPos, normal);
		result += light.intensity.rgb * (cosine * visibility / distanceSquared);
	}
	return result;
}
