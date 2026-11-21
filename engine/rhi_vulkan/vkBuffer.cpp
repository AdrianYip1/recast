#include "vkBuffer.hpp"

foto::Buffer::Buffer(Device& device, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties) :
	_device(device.getDevice()), _size(size),
	buffer(createBuffer(device, size, usage)),
	memory(allocateMemory(device, buffer, memoryProperties))
{

};

foto::Buffer::~Buffer() {
	vkDestroyBuffer(_device, buffer, nullptr);
	vkFreeMemory(_device, memory, nullptr);
};

std::unique_ptr<foto::Buffer> foto::Buffer::createDeviceLocal(Device& device, VkCommandPool commandPool,
															  const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
	// CPU writable staging buffer
	Buffer staging(device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	staging.upload(data, size);

	// GPU only buffer, copied into from staging
	auto result = std::make_unique<Buffer>(device, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
										   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	result->copyFrom(device, commandPool, staging);

	// staging dtor here
	return result;
}

void foto::Buffer::upload(const void* data, VkDeviceSize dataSize) {
	if (dataSize > _size) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> upload(): data larger than buffer");
	}

	void* mapped = nullptr;
	if (vkMapMemory(_device, memory, 0, dataSize, 0, &mapped) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> upload()");
	}
	memcpy(mapped, data, static_cast<size>(dataSize));
	vkUnmapMemory(_device, memory);
}

VkBuffer foto::Buffer::createBuffer(Device& device, VkDeviceSize size, VkBufferUsageFlags usage) {
	VkBufferCreateInfo bufferCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE
	};

	VkBuffer buffer = VK_NULL_HANDLE;

	if (vkCreateBuffer(device.getDevice(), &bufferCreateInfo, nullptr, &buffer) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> createBuffer()");
	}

	return buffer;
}

VkDeviceMemory foto::Buffer::allocateMemory(Device& device, VkBuffer buffer, VkMemoryPropertyFlags memoryProperties) {
	VkMemoryRequirements memReq;
	vkGetBufferMemoryRequirements(device.getDevice(), buffer, &memReq);

	VkMemoryAllocateInfo allocInfo{
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memReq.size,
		.memoryTypeIndex = findMemoryType(device.getPhysicalDevice(), memReq.memoryTypeBits, memoryProperties)
	};

	VkDeviceMemory memory = VK_NULL_HANDLE;

	if (vkAllocateMemory(device.getDevice(), &allocInfo, nullptr, &memory) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> allocateMemory()");
	}

	vkBindBufferMemory(device.getDevice(), buffer, memory, 0);

	return memory;
}

foto::u32 foto::Buffer::findMemoryType(VkPhysicalDevice physDevice, u32 typeFilter, VkMemoryPropertyFlags properties) {
	// Available types of memory
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physDevice, &memProperties);

	for (u32 i = 0; i < memProperties.memoryTypeCount; i++) {
		if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("ERROR: vkBuffer.cpp -> findMemoryType()");
}

void foto::Buffer::copyFrom(Device& device, VkCommandPool commandPool, Buffer& src) {
	if (src.getSize() > _size) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> copyFrom(): src larger than buffer");
	}

	VkCommandBufferAllocateInfo commandBufferAllocInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = commandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};

	VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
	if (vkAllocateCommandBuffers(_device, &commandBufferAllocInfo, &commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> copyFrom() allocate");
	}

	VkCommandBufferBeginInfo beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> copyFrom() begin");
	}

	VkBufferCopy copyRegion = {
		.size = src.getSize(),
	};

	vkCmdCopyBuffer(commandBuffer, src.getBuffer(), buffer, 1, &copyRegion);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> copyFrom() end");
	}

	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer
	};

	VkQueue queue = device.getGraphicsQ();
	if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> copyFrom() submit");
	}
	vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(_device, commandPool, 1, &commandBuffer);
}