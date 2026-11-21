#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNormal;
layout(location=2) in vec2 inUV;

layout(location=0) out vec3 fragNormal;
layout(location=1) out vec2 fragUV;
layout(location=2) out vec3 fragWorldPos;

void main() {
	vec4 worldPos = push.model * vec4(inPos, 1.0);
	gl_Position = frame.proj * frame.view * worldPos;

	// Inverse-transpose keeps normals perpendicular under non-uniform scale
	fragNormal = transpose(inverse(mat3(push.model))) * inNormal;
	fragUV = inUV;
	fragWorldPos = worldPos.xyz;
}