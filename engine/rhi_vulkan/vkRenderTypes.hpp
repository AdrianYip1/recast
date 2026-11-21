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

	// One shadow atlas row per light. Point lights: a cube map, 6 faces (+X -X +Y -Y +Z -Z) side by side.
	// Directional lights: one orthographic map in face 0
	struct GpuLight {
		alignas(16) enginemath::Vec4 position;  // xyz, w = 1 if it casts shadows
		alignas(16) enginemath::Vec4 intensity; // rgb, w = 0 point, 1 directional, 2 area. Point: irradiance = intensity * cos / distance^2, directional: intensity * cos, area: radiance
		alignas(16) enginemath::Vec4 direction; // xyz = the way a directional light travels (area: half its width), w = Light::size
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
		alignas(16) enginemath::Vec4 exposure;  // x = linear multiplier (2^EV), y = camera grain, z = camera blur (px), w = 1 to apply them
		alignas(16) enginemath::Vec4 whiteBalance; // rgb gains, w = path traced photo's texture slot (camera matching)
		alignas(16) enginemath::Vec4 ambient;   // rgb irradiance from everything that isn't a point light
		alignas(16) enginemath::Vec4 viewport;  // 3D view rect in framebuffer pixels: x, y, width, height
		alignas(16) u32 lightInfo[4] = {};      // light count, static shadow atlas texture, dynamic shadow atlas texture, face resolution
		GpuLight lights[MAX_LIGHTS];
	};

	// Per-draw push constants ("Push" in the shaders)
	struct DrawPush {
		enginemath::Mat4 model;
		u32 materialIndex;
		u32 _pad[3] = {};
		enginemath::Vec4 tint = { 1.0f, 1.0f, 1.0f, 1.0f }; // multiplies the lighting of the lit shader
	};
	static_assert(sizeof(DrawPush) == 96);

	inline constexpr VkPushConstantRange DRAW_PUSH_RANGE = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(DrawPush)
	};

	inline constexpr u32 NO_MATERIAL_OVERRIDE = UINT32_MAX;

	inline constexpr u32 MAX_TEXTURES = 1024; // bindless texture array size, smaller on GPUs that can't bind this many
	inline constexpr u32 MIN_TEXTURES = 16;
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
		f32 specular = 1.0f; // KHR_materials_specular factor: 0 = no gloss at all (a purely diffuse surface)
		f32 transmission = 0.0f; // KHR_materials_transmission: 1 = clear glass (path tracing only)
		f32 ior = 1.5f;          // KHR_materials_ior
		u32 maskId = 0; // path tracing: the cutout (owner image value) this room surface belongs to, 0 = none
		f32 _pad = 0.0f;
	};
	static_assert(sizeof(Material) == 64);

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
		enginemath::Vec3 tint = enginemath::Vec3(1.0f, 1.0f, 1.0f);
	};

	// Path tracing: static casters are the photo's surfaces, dynamic casters the inserted objects
	inline constexpr u32 MAX_TRACE_INSTANCES = 256;
	inline constexpr u32 MAX_TRACE_GEOMETRIES = 1024;
	inline constexpr u32 MAX_TRACE_SAMPLES = 1024;

	// One per (instance, submesh): set 1, binding 3 ("Geometries" in pathtrace.comp)
	struct GpuGeometry {
		VkDeviceAddress vertexAddress;
		VkDeviceAddress indexAddress;
		u32 firstIndex;
		u32 materialIndex;
		u32 dynamic;
		u32 _pad = 0;
	};
	static_assert(sizeof(GpuGeometry) == 32);

	inline constexpr u32 CAUSTIC_PHOTONS = 65536; // per frame

	struct TracePush {
		u32 sampleIndex;
		u32 photoTexture;
		u32 trace; // 0 = only show the accumulated result
		u32 photonCount = CAUSTIC_PHOTONS;
		enginemath::Vec4 objectSphere; // bounding sphere of the dynamic casters, w = radius (0 = none)
		u32 ownerTexture = 0; // per photo pixel: which cutout it shows (Material::maskId), 0 = no owner image
		u32 _pad[3] = {};
	};
	static_assert(sizeof(TracePush) == 48);

	inline constexpr VkPushConstantRange TRACE_PUSH_RANGE = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(TracePush)
	};

	inline constexpr u32 DENOISE_PASSES = 4;

	struct DenoisePush {
		u32 pass;
		u32 passCount = DENOISE_PASSES;
		u32 samples;
		u32 _pad = 0;
	};

	inline constexpr VkPushConstantRange DENOISE_PUSH_RANGE = {
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(DenoisePush)
	};

	enum class LightType : u32 { Point, Directional, Area };

	// Area: a vertical rectangle (a window), position = its centre, direction = half its width along the wall (not
	// normalized), size = half its height, intensity = radiance. It shines towards cross(direction, up). The raster path
	// treats it as a point at its centre
	struct Light {
		LightType type = LightType::Point;
		enginemath::Vec3 position;
		enginemath::Vec3 direction = enginemath::Vec3(0.0f, -1.0f, 0.0f);
		enginemath::Vec3 intensity;
		f32 size = 0.0f; // point: radius in metres, directional: angular radius in radians, area: half height. Path tracing only (soft shadows)
		bool castsShadows = true;
	};

	// View-projections of the 6 cube faces around a point light, in the atlas face order
	void shadowFaceMatrices(const enginemath::Vec3& position, enginemath::Mat4 out[6]);

	// Orthographic view-projection for a directional light covering a sphere of `radius` around `center`
	enginemath::Mat4 shadowDirectionalMatrix(const enginemath::Vec3& direction, const enginemath::Vec3& center, f32 radius);
} // namespace foto
