#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "lighting.glsl"

layout(location=0) in vec3 fragNormal;
layout(location=1) in vec2 fragUV;
layout(location=2) in vec3 fragWorldPos;

layout(location=0) out vec4 outColor;

// Differential rendering for diffuse surfaces: albedo cancels, so the photo is scaled by how much the
// dynamic casters change the irradiance (shadows from the inserted object)
void main() {
	Material material = materials[push.materialIndex];
	vec2 screenUV = (gl_FragCoord.xy - frame.viewport.xy) / frame.viewport.zw;
	vec3 photo = texture(textures[material.baseColorTexture], screenUV).rgb;

	vec3 n = normalize(fragNormal);
	if (!gl_FrontFacing) n = -n;

	vec3 withObject = irradiance(fragWorldPos, n, true);
	vec3 withoutObject = irradiance(fragWorldPos, n, false);
	outColor = vec4(photo * withObject / max(withoutObject, vec3(1e-4)), 1.0);
}
