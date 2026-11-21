#pragma once

#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"
#include "vkBuffer.hpp"
#include "vkVertex.hpp"

namespace foto {
	// A range of the mesh's index buffer drawn with one material
	struct Submesh {
		u32 firstIndex = 0;
		u32 indexCount = 0;
		u32 materialIndex = 0;
	};

	// Owns a device-local vertex + index buffer and the submeshes that split it by material
	class Mesh {
	public:
		// Empty submeshes = one submesh covering every index with material 0
		Mesh(Device& device, VkCommandPool commandPool, std::span<const Vertex> vertices,
			 std::span<const u32> indices, std::vector<Submesh> submeshes = {});
		Mesh(const Mesh& other) = delete;
		Mesh& operator=(const Mesh& other) = delete;

		// Records binding the vertex + index buffers (binding 0, 32-bit indices)
		void bind(VkCommandBuffer commandBuffer);

		Buffer& getVertexBuffer() { return *vertexBuffer; };
		Buffer& getIndexBuffer() { return *indexBuffer; };
		u32 getVertexCount() { return vertexCount; };
		u32 getIndexCount() { return indexCount; };
		std::vector<Submesh>& getSubmeshes() { return submeshes; };
		const enginemath::Vec3& getBoundsMin() { return boundsMin; };
		const enginemath::Vec3& getBoundsMax() { return boundsMax; };

	private:
		u32 vertexCount = 0;
		u32 indexCount = 0;
		std::unique_ptr<Buffer> vertexBuffer;
		std::unique_ptr<Buffer> indexBuffer;
		std::vector<Submesh> submeshes;
		enginemath::Vec3 boundsMin;
		enginemath::Vec3 boundsMax;

		static u32 nonEmptyCount(size count);
	};
} // namespace foto
