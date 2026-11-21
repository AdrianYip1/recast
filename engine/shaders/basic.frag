#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location=0) in vec3 fragNormal;
layout(location=1) in vec2 fragUV;
layout(location=2) in vec3 fragWorldPos;

layout(set=0, binding=1) uniform sampler2D texSampler;

layout(location=0) out vec4 outColor;

void main() {
	vec4 albedo = texture(texSampler, fragUV);

	// Simple preview lighting: ambient + one directional light (two-sided so back faces aren't black)
	vec3 n = normalize(fragNormal);
	float diffuse = abs(dot(n, -frame.lightDir.xyz));
	vec3 color = albedo.rgb * (0.35 + 0.65 * diffuse);

	outColor = vec4(color, albedo.a);
}