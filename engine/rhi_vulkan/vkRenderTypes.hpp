#pragma once
#include <enginemath/enginemath.hpp>

#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

// Data shared between C++ and the shaders. Layouts must match the GLSL blocks exactly
// (std140 for uniforms, std430 for storage buffers / push constants).

namespace foto {
	class Mesh;

	// Per-frame uniforms: set 0, binding 0 ("FrameUbo" in the shaders)
	struct FrameUbo {
		alignas(16) enginemath::Mat4 view;
		alignas(16) enginemath::Mat4 proj;
		alignas(16) enginemath::Vec4 cameraPos; // xyz = world position
		alignas(16) enginemath::Vec4 lightDir;  // xyz = direction the light travels (normalized)
		alignas(16) enginemath::Vec4 exposure;  // x = linear multiplier (2^EV)
		alignas(16) enginemath::Vec4 whiteBalance; // rgb gains
	};

	// Per-draw push constants ("Push" in the shaders)
	struct DrawPush {
		enginemath::Mat4 model;
		u32 materialIndex;
	};

	inline constexpr VkPushConstantRange DRAW_PUSH_RANGE = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(DrawPush)
	};

	inline constexpr u32 NO_MATERIAL_OVERRIDE = UINT32_MAX;

	inline constexpr u32 MAX_TEXTURES = 1024;
	inline constexpr u32 MAX_MATERIALS = 1024;
	inline constexpr u32 DEFAULT_TEXTURE = 0;
	inline constexpr u32 DEFAULT_MATERIAL = 0;

	// set 0, binding 1 (std430 storage buffer)
	struct Material {
		alignas(16) enginemath::Vec4 baseColor = { 1.0f, 1.0f, 1.0f, 1.0f };
		alignas(16) enginemath::Vec4 emissive = { 0.0f, 0.0f, 0.0f, 0.0f };
		f32 roughness = 1.0f;
		f32 metallic = 0.0f;
		u32 baseColorTexture = DEFAULT_TEXTURE;
		u32 _pad = 0;
	};
	static_assert(sizeof(Material) == 48);

	// One mesh drawn with one transform. Every submesh uses its own material unless overridden
	struct DrawItem {
		Mesh* mesh = nullptr;
		enginemath::Mat4 transform;
		u32 materialOverride = NO_MATERIAL_OVERRIDE;
	};
} // namespace foto
