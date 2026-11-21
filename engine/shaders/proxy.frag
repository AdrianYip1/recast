#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location=0) in vec3 fragNormal;
layout(location=1) in vec2 fragUV;
layout(location=2) in vec3 fragWorldPos;

layout(location=0) out vec4 outColor;

const float GRID_SPACING = 0.5;
const vec3 GRID_COLOR = vec3(1.0, 0.8, 0.1);

float gridLine(float coordinate) {
	float distanceToLine = abs(fract(coordinate / GRID_SPACING + 0.5) - 0.5) * GRID_SPACING;
	return 1.0 - smoothstep(0.0, fwidth(coordinate) * 1.5, distanceToLine);
}

void main() {
	Material material = materials[push.materialIndex];
	vec3 n = abs(normalize(fragNormal));

	float line = 0.0;
	if (n.x < 0.9) line = max(line, gridLine(fragWorldPos.x));
	if (n.y < 0.9) line = max(line, gridLine(fragWorldPos.y));
	if (n.z < 0.9) line = max(line, gridLine(fragWorldPos.z));

	outColor = vec4(mix(material.baseColor.rgb, GRID_COLOR, line), mix(material.baseColor.a, 1.0, line));
}
