#include "vkImage.hpp"
#include "vkBuffer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

foto::Image::Image(Device& device, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect) :
	_device(device.getDevice()), _extent(extent), _format(format),
	image(createImage(device, extent, format, usage)),
	memory(allocateMemory(device, image)),
	view(createImageView(device, image, format, aspect))
{

};

foto::Image::~Image() {
	vkDestroyImageView(_device, view, nullptr);
	vkDestroyImage(_device, image, nullptr);
	vkFreeMemory(_device, memory, nullptr);
};

std::unique_ptr<foto::Image> foto::Image::createTexture(Device& device, VkCommandPool commandPool, const void* pixels,
														VkDeviceSize size, VkExtent2D extent, VkFormat format) {
	// CPU writable staging buffer
	Buffer staging(device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	staging.upload(pixels, size);

	// GPU only image, copied into from staging
	auto texture = std::make_unique<Image>(device, extent, format,
										   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

	device.immediateSubmit(commandPool, [&](VkCommandBuffer commandBuffer) {
		// UNDEFINED -> TRANSFER_DST: contents don't matter yet, we're about to overwrite them
		transitionLayout(commandBuffer, texture->getImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		// Tightly packed pixels (rowLength/imageHeight 0) into the whole image
		VkBufferImageCopy copyRegion = {
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
			.imageOffset = {0, 0, 0},
			.imageExtent = {extent.width, extent.height, 1}
		};
		vkCmdCopyBufferToImage(commandBuffer, staging.getBuffer(), texture->getImage(),
							   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

		// TRANSFER_DST -> SHADER_READ_ONLY: ready to be sampled
		transitionLayout(commandBuffer, texture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	});

	// staging dtor here
	return texture;
}

std::unique_ptr<foto::Image> foto::Image::loadTexture(Device& device, VkCommandPool commandPool, const std::string& filePath) {
	int width = 0, height = 0, channels = 0;

	// Force 4 channels so every texture is RGBA8 regardless of the file
	stbi_uc* pixels = stbi_load(filePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
	if (!pixels) {
		throw std::runtime_error("ERROR: vkImage.cpp -> loadTexture(): " + filePath + " (" + stbi_failure_reason() + ")");
	}

	VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * 4;
	VkExtent2D extent = { static_cast<u32>(width), static_cast<u32>(height) };

	std::unique_ptr<Image> texture;
	try {
		texture = createTexture(device, commandPool, pixels, size, extent, VK_FORMAT_R8G8B8A8_SRGB);
	}
	catch (...) {
		stbi_image_free(pixels);
		throw;
	}

	stbi_image_free(pixels);
	return texture;
}

void foto::Image::transitionLayout(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout,
								   VkImageLayout newLayout, VkImageAspectFlags aspect) {
	VkImageMemoryBarrier2 barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,

		.oldLayout = oldLayout,
		.newLayout = newLayout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS},
	};

	VkDependencyInfo dependency = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier,
	};
	vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

VkImage foto::Image::createImage(Device& device, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage) {
	VkImageCreateInfo imageCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = {extent.width, extent.height, 1},
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
	};

	VkImage image = VK_NULL_HANDLE;

	if (vkCreateImage(device.getDevice(), &imageCreateInfo, nullptr, &image) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkImage.cpp -> createImage()");
	}

	return image;
}

VkDeviceMemory foto::Image::allocateMemory(Device& device, VkImage image) {
	VkMemoryRequirements memReq;
	vkGetImageMemoryRequirements(device.getDevice(), image, &memReq);

	VkMemoryAllocateInfo allocInfo = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = memReq.size,
		.memoryTypeIndex = device.findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
	};

	VkDeviceMemory memory = VK_NULL_HANDLE;

	if (vkAllocateMemory(device.getDevice(), &allocInfo, nullptr, &memory) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkImage.cpp -> allocateMemory()");
	}

	vkBindImageMemory(device.getDevice(), image, memory, 0);

	return memory;
}

VkImageView foto::Image::createImageView(Device& device, VkImage image, VkFormat format, VkImageAspectFlags aspect) {
	VkImageViewCreateInfo viewCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {aspect, 0, 1, 0, 1}
	};

	VkImageView view = VK_NULL_HANDLE;

	if (vkCreateImageView(device.getDevice(), &viewCreateInfo, nullptr, &view) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkImage.cpp -> createImageView()");
	}

	return view;
}


foto::Sampler::Sampler(Device& device, VkFilter filter, VkSamplerAddressMode addressMode) :
	_device(device.getDevice()), sampler(createSampler(device, filter, addressMode))
{

};

foto::Sampler::~Sampler() {
	vkDestroySampler(_device, sampler, nullptr);
};

VkSampler foto::Sampler::createSampler(Device& device, VkFilter filter, VkSamplerAddressMode addressMode) {
	VkSamplerCreateInfo samplerCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = filter,
		.minFilter = filter,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = addressMode,
		.addressModeV = addressMode,
		.addressModeW = addressMode,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE, // needs the samplerAnisotropy device feature
		.maxAnisotropy = 1.0f,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE
	};

	VkSampler sampler = VK_NULL_HANDLE;

	if (vkCreateSampler(device.getDevice(), &samplerCreateInfo, nullptr, &sampler) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkImage.cpp -> createSampler()");
	}

	return sampler;
}
