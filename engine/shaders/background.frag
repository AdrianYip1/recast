#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location=0) in vec2 fragUV;

layout(location=0) out vec4 outColor;

void main() {
	Material material = materials[push.materialIndex];
	outColor = material.baseColor * texture(textures[material.baseColorTexture], fragUV);
}
