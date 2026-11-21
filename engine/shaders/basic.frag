#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "lighting.glsl"

layout(location=0) in vec3 fragNormal;
layout(location=1) in vec2 fragUV;
layout(location=2) in vec3 fragWorldPos;

layout(location=0) out vec4 outColor;

void main() {
	Material material = materials[push.materialIndex];
	vec4 albedo = material.baseColor * texture(textures[material.baseColorTexture], fragUV);

	vec3 n = normalize(fragNormal);
	if (!gl_FrontFacing) n = -n;

	vec3 color = albedo.rgb * irradiance(fragWorldPos, n, true) + material.emissive.rgb;
	outColor = vec4(color, albedo.a);
}
