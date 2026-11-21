#pragma once

#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include <engine/rhi_vulkan/vkVertex.hpp>
#include <engine/rhi_vulkan/vkMesh.hpp>
#include <engine/rhi_vulkan/vkRenderTypes.hpp>

namespace foto {
	inline constexpr u32 NO_TEXTURE = UINT32_MAX;

	struct ImageData {
		std::vector<u8> rgba;
		u32 width = 0;
		u32 height = 0;
	};

	// Node transforms are baked into the vertices. Submesh::materialIndex indexes `materials`,
	// Material::baseColorTexture indexes `images` (or is NO_TEXTURE)
	struct ModelData {
		std::vector<Vertex> vertices;
		std::vector<u32> indices;
		std::vector<Submesh> submeshes;
		std::vector<Material> materials;
		std::vector<ImageData> images;
	};

	// .gltf (external or embedded buffers/images) and .glb
	ModelData loadGltf(const std::string& filePath);
} // namespace foto
