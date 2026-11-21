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
	vec3 n = normalize(fragNormal);
	if (!gl_FrontFacing) n = -n;
	outColor = vec4(material.baseColor.rgb * irradiance(fragWorldPos, n, true), 1.0);
}
