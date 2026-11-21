// Must match engine/rhi_vulkan/vkRenderTypes.hpp

#define MAX_TEXTURES 1024

layout(set = 0, binding = 0) uniform FrameUbo {
	mat4 view;
	mat4 proj;
	vec4 cameraPos;
	vec4 lightDir;
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

layout(push_constant) uniform Push {
	mat4 model;
	uint materialIndex;
} push;
