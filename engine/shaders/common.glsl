// Must match engine/rhi_vulkan/vkRenderTypes.hpp

#define MAX_TEXTURES 1024
#define MAX_LIGHTS 4

struct Light {
	vec4 position;
	vec4 intensity;
	vec4 direction;
	mat4 faceViewProj[6];
};

layout(set = 0, binding = 0) uniform FrameUbo {
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
	vec4 lightDir;
	vec4 exposure;
	vec4 whiteBalance;
	vec4 ambient;
	vec4 viewport;
	uvec4 lightInfo;
	Light lights[MAX_LIGHTS];
} frame;

struct Material {
	vec4 baseColor;
	vec4 emissive;
	float roughness;
	float metallic;
	uint baseColorTexture;
	uint pad;
};

layout(std430, set = 0, binding = 1) readonly buffer Materials {
	Material materials[];
};

layout(set = 0, binding = 2) uniform sampler2D textures[MAX_TEXTURES];

#ifndef CUSTOM_PUSH
layout(push_constant) uniform Push {
	mat4 model;
	uint materialIndex;
	vec4 tint;
} push;
#endif
