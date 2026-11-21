#include "vkMesh.hpp"

foto::Mesh::Mesh(Device& device, VkCommandPool commandPool, std::span<const Vertex> vertices,
				 std::span<const u32> indices, std::vector<Submesh> submeshes) :
	// Checked here (not in the body) so empty data throws before any zero-sized buffer is created
	vertexCount(nonEmptyCount(vertices.size())),
	indexCount(nonEmptyCount(indices.size())),
	vertexBuffer(Buffer::createDeviceLocal(device, commandPool, vertices.data(), vertices.size_bytes(),
										   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingUsage(device))),
	indexBuffer(Buffer::createDeviceLocal(device, commandPool, indices.data(), indices.size_bytes(),
										  VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingUsage(device))),
	submeshes(std::move(submeshes))
{
	boundsMin = boundsMax = vertices[0].pos;
	for (const Vertex& vertex : vertices) {
		for (int a = 0; a < 3; a++) {
			boundsMin[a] = std::min(boundsMin[a], vertex.pos[a]);
			boundsMax[a] = std::max(boundsMax[a], vertex.pos[a]);
		}
	}

	if (this->submeshes.empty()) {
		this->submeshes.push_back({ .firstIndex = 0, .indexCount = indexCount, .materialIndex = 0 });
	}

	if (device.supportsRayTracing()) {
		bottomLevel = AccelerationStructure::buildBottomLevel(device, commandPool, *vertexBuffer, vertexCount, *indexBuffer, this->submeshes);
	}
};

foto::Mesh::~Mesh() = default;

VkBufferUsageFlags foto::Mesh::rayTracingUsage(Device& device) {
	if (!device.supportsRayTracing()) return 0;
	return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
}

foto::u32 foto::Mesh::nonEmptyCount(size count) {
	if (count == 0) {
		throw std::runtime_error("ERROR: vkMesh.cpp -> Mesh(): empty vertices or indices");
	}
	return static_cast<u32>(count);
}

void foto::Mesh::bind(VkCommandBuffer commandBuffer) {
	VkBuffer buffers[] = { vertexBuffer->getBuffer() };
	VkDeviceSize offsets[] = { 0 };

	vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, offsets);
	vkCmdBindIndexBuffer(commandBuffer, indexBuffer->getBuffer(), 0, VK_INDEX_TYPE_UINT32);
}
