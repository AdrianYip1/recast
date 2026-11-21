// Must match engine/rhi_vulkan/vkRenderTypes.hpp

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
	vec4 exposure; // x = 2^EV, y = camera grain, z = camera blur in pixels, w = 1 to apply both to the object (alpha)
	vec4 whiteBalance; // w = photo texture slot while path tracing
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
	float specular;
	float transmission;
	float ior;
	uint maskId; // the cutout (owner image value) this room surface belongs to, 0 = none
	float pad1;
};

layout(std430, set = 0, binding = 1) readonly buffer Materials {
	Material materials[];
};

// Set per GPU by the pipelines (1024 unless the GPU can't bind that many)
layout(constant_id = 0) const uint TEXTURE_CAPACITY = 1024;
layout(set = 0, binding = 2) uniform sampler2D textures[TEXTURE_CAPACITY];

#ifndef CUSTOM_PUSH
layout(push_constant) uniform Push {
	mat4 model;
	uint materialIndex;
	vec4 tint;
} push;
#endif
