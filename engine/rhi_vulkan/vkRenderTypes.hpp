#pragma once
#include <enginemath/enginemath.hpp>

#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

// Data shared between C++ and the shaders. Layouts must match the GLSL blocks exactly
// (std140 for uniforms, std430 for storage buffers / push constants).

namespace foto {
	class Mesh;
	class Pipeline;

	inline constexpr u32 MAX_LIGHTS = 4;
	inline constexpr u32 SHADOW_FACE_RESOLUTION = 512;

	// Point light with a cube shadow map: 6 faces (+X -X +Y -Y +Z -Z) side by side in a shadow atlas row
	struct GpuLight {
		alignas(16) enginemath::Vec4 position;  // xyz, w = 1 if it casts shadows
		alignas(16) enginemath::Vec4 intensity; // rgb radiant intensity (irradiance = intensity * cos / distance^2)
		alignas(16) enginemath::Mat4 faceViewProj[6] = {
			enginemath::Mat4::identity(), enginemath::Mat4::identity(), enginemath::Mat4::identity(),
			enginemath::Mat4::identity(), enginemath::Mat4::identity(), enginemath::Mat4::identity()
		};
	};

	// Per-frame uniforms: set 0, binding 0 ("FrameUbo" in the shaders)
	struct FrameUbo {
		alignas(16) enginemath::Mat4 view;
		alignas(16) enginemath::Mat4 proj;
		alignas(16) enginemath::Vec4 cameraPos; // xyz = world position
		alignas(16) enginemath::Vec4 lightDir;  // xyz = direction the light travels, used when there are no point lights
		alignas(16) enginemath::Vec4 exposure;  // x = linear multiplier (2^EV)
		alignas(16) enginemath::Vec4 whiteBalance; // rgb gains
		alignas(16) enginemath::Vec4 ambient;   // rgb irradiance from everything that isn't a point light
		alignas(16) enginemath::Vec4 viewport;  // 3D view rect in framebuffer pixels: x, y, width, height
		alignas(16) u32 lightInfo[4] = {};      // light count, static shadow atlas texture, dynamic shadow atlas texture, face resolution
		GpuLight lights[MAX_LIGHTS];
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

	// One mesh drawn with one transform. Every submesh uses its own material unless overridden.
	// pipeline = nullptr uses the default scene pipeline. mesh = nullptr draws a fullscreen triangle (needs a pipeline).
	// Static casters (e.g. the room) and dynamic casters (e.g. a moving object) go into separate shadow atlases
	// so shaders can tell the two kinds of occlusion apart
	enum class ShadowCaster : u32 { None, Static, Dynamic };

	struct DrawItem {
		Mesh* mesh = nullptr;
		enginemath::Mat4 transform;
		u32 materialOverride = NO_MATERIAL_OVERRIDE;
		Pipeline* pipeline = nullptr;
		ShadowCaster shadowCaster = ShadowCaster::None;
	};

	struct PointLight {
		enginemath::Vec3 position;
		enginemath::Vec3 intensity;
		bool castsShadows = true;
	};

	// View-projections of the 6 cube faces around a light, in the atlas face order
	void shadowFaceMatrices(const enginemath::Vec3& position, enginemath::Mat4 out[6]);
} // namespace foto
