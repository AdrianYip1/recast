#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location=0) in vec2 fragUV;

layout(location=0) out vec4 outColor;

void main() {
	vec3 hdr = texture(textures[materials[push.materialIndex].baseColorTexture], fragUV).rgb;
	vec3 color = hdr * frame.exposure.x * frame.whiteBalance.rgb;
	outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
