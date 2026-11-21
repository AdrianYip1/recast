// Shared by every shader: #include "common.glsl" (needs GL_GOOGLE_include_directive)
// Layouts must match engine/rhi_vulkan/vkRenderTypes.hpp

layout(set = 0, binding = 0) uniform FrameUbo {
	mat4 view;
	mat4 proj;
	vec4 cameraPos; // xyz = world position
	vec4 lightDir;  // xyz = direction the light travels (normalized)
} frame;

layout(push_constant) uniform Push {
	mat4 model;
	uint materialIndex;
} push;
