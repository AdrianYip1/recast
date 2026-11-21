#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNormal;
layout(location=2) in vec2 inUV;

void main() {
	uint lightIndex = push.materialIndex / 6u;
	uint face = push.materialIndex % 6u;
	gl_Position = frame.lights[lightIndex].faceViewProj[face] * push.model * vec4(inPos, 1.0);
}
