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

void foto::Buffer::upload(const void* data, VkDeviceSize dataSize, VkDeviceSize offset) {
	if (offset + dataSize > _size) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> upload(): data larger than buffer");
	}

	void* mapped = nullptr;
	if (vkMapMemory(_device, memory, offset, dataSize, 0, &mapped) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> upload()");
	}
	memcpy(mapped, data, static_cast<size>(dataSize));
	vkUnmapMemory(_device, memory);
}

void foto::Buffer::download(void* out, VkDeviceSize dataSize, VkDeviceSize offset) {
	if (offset + dataSize > _size) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> download(): range larger than buffer");
	}

	void* mapped = nullptr;
	if (vkMapMemory(_device, memory, offset, dataSize, 0, &mapped) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> download()");
	}
	memcpy(out, mapped, static_cast<size>(dataSize));
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
		.memoryTypeIndex = device.findMemoryType(memReq.memoryTypeBits, memoryProperties)
	};

	VkDeviceMemory memory = VK_NULL_HANDLE;

	if (vkAllocateMemory(device.getDevice(), &allocInfo, nullptr, &memory) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> allocateMemory()");
	}

	vkBindBufferMemory(device.getDevice(), buffer, memory, 0);

	return memory;
}

void foto::Buffer::copyFrom(Device& device, VkCommandPool commandPool, Buffer& src) {
	if (src.getSize() > _size) {
		throw std::runtime_error("ERROR: vkBuffer.cpp -> copyFrom(): src larger than buffer");
	}

	device.immediateSubmit(commandPool, [&](VkCommandBuffer commandBuffer) {
		VkBufferCopy copyRegion = {
			.size = src.getSize(),
		};
		vkCmdCopyBuffer(commandBuffer, src.getBuffer(), buffer, 1, &copyRegion);
	});
}