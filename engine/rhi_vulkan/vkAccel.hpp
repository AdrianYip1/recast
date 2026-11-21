#pragma once

#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"
#include "vkBuffer.hpp"

namespace foto {
	struct Submesh;

	// Owns a VkAccelerationStructureKHR and the buffer it lives in
	class AccelerationStructure {
	public:
		AccelerationStructure(Device& device, VkAccelerationStructureTypeKHR type, VkDeviceSize size);
		~AccelerationStructure();
		AccelerationStructure(const AccelerationStructure& other) = delete;
		AccelerationStructure& operator=(const AccelerationStructure& other) = delete;

		// One geometry per submesh, so ray queries report the submesh index as the geometry index.
		// The buffers need ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY and SHADER_DEVICE_ADDRESS usage
		static std::unique_ptr<AccelerationStructure> buildBottomLevel(Device& device, VkCommandPool commandPool, Buffer& vertices,
																	   u32 vertexCount, Buffer& indices, std::span<const Submesh> submeshes);

		VkAccelerationStructureKHR getHandle() { return handle; };
		VkDeviceAddress getAddress() { return address; };

	private:
		Device& device;
		std::unique_ptr<Buffer> buffer;
		VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
		VkDeviceAddress address = 0;
	};

	// Top level structure rebuilt from scratch each time it's recorded, up to maxInstances instances
	class TopLevelAccelerationStructure {
	public:
		TopLevelAccelerationStructure(Device& device, u32 maxInstances);

		// Uploads the instances and records the build. Don't call while a submitted build of this structure is still running.
		// Readers need a barrier from ACCELERATION_STRUCTURE_BUILD / WRITE afterwards
		void build(VkCommandBuffer commandBuffer, std::span<const VkAccelerationStructureInstanceKHR> instances);

		VkAccelerationStructureKHR getHandle() { return structure->getHandle(); };
		u32 getMaxInstances() { return maxInstances; };

	private:
		Device& device;
		u32 maxInstances = 0;
		std::unique_ptr<Buffer> instanceBuffer;
		std::unique_ptr<Buffer> scratch;
		std::unique_ptr<AccelerationStructure> structure;
	};
} // namespace foto
