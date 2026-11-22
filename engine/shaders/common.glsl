// Must match engine/rhi_vulkan/vkRenderTypes.hpp

#define MAX_LIGHTS 4

struct Light {
	vec4 position;
	vec4 intensity;
	vec4 direction;
	mat4 faceViewProj[6];
};

// intensity.w: 0 = point, 1 = directional, 2 = area: a vertical rectangle (a window), position = its centre,
// direction.xyz = half its width along the wall, direction.w = half its height, intensity = radiance,
// shining towards cross(direction.xyz, up)
bool isDirectional(Light light) { return abs(light.intensity.w - 1.0) < 0.5; }
bool isArea(Light light) { return light.intensity.w > 1.5; }
vec3 areaNormal(Light light) { return normalize(cross(light.direction.xyz, vec3(0.0, 1.0, 0.0))); }
float areaSize(Light light) { return 4.0 * length(light.direction.xyz) * light.direction.w; }
// A point at the light's position standing in for it (raster path, photons): area lights send radiance * area
vec3 pointIntensity(Light light) { return isArea(light) ? light.intensity.rgb * areaSize(light) : light.intensity.rgb; }

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
	vec4 absorption; // light lost per metre inside glass (rgb)
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
