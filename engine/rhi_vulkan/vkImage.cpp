#include "vkImage.hpp"
#include "vkBuffer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

foto::Image::Image(Device& device, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage,
				   VkImageAspectFlags aspect, u32 mipLevels) :
	_device(device.getDevice()), _extent(extent), _format(format), _mipLevels(mipLevels),
	image(createImage(device, extent, format, usage, mipLevels)),
	memory(allocateMemory(device, image)),
	view(createImageView(device, image, format, aspect, mipLevels))
{

};

foto::Image::~Image() {
	vkDestroyImageView(_device, view, nullptr);
	vkDestroyImage(_device, image, nullptr);
	vkFreeMemory(_device, memory, nullptr);
};

std::unique_ptr<foto::Image> foto::Image::createTexture(Device& device, VkCommandPool commandPool, const void* pixels,
														VkDeviceSize size, VkExtent2D extent, VkFormat format,
														bool generateMips) {
	// Mips are made by linearly blitting each level from the one above: the format must support
	// being a blit source, a blit destination, and linear filtering (all for optimal tiling)
	if (generateMips) {
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(device.getPhysicalDevice(), format, &formatProperties);

		constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
												  VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
		generateMips = (formatProperties.optimalTilingFeatures & required) == required;
	}
	u32 mipLevels = generateMips ? fullMipCount(extent) : 1;

	// CPU writable staging buffer
	Buffer staging(device, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	staging.upload(pixels, size);

	// GPU only image, copied into from staging. TRANSFER_SRC: each mip level is blitted from the previous one
	VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	if (mipLevels > 1) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

	auto texture = std::make_unique<Image>(device, extent, format, usage, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);

	device.immediateSubmit(commandPool, [&](VkCommandBuffer commandBuffer) {
		// UNDEFINED -> TRANSFER_DST (all levels): contents don't matter yet, we're about to overwrite them
		transitionLayout(commandBuffer, texture->getImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		// Tightly packed pixels (rowLength/imageHeight 0) into level 0
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

		if (mipLevels > 1) {
			// Fills levels 1..N from level 0 and leaves them all SHADER_READ_ONLY
			generateMipmaps(commandBuffer, texture->getImage(), extent, mipLevels);
		}
		else {
			// TRANSFER_DST -> SHADER_READ_ONLY: ready to be sampled
			transitionLayout(commandBuffer, texture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
	});

	// staging dtor here
	return texture;
}

std::unique_ptr<foto::Image> foto::Image::loadTexture(Device& device, VkCommandPool commandPool, const std::string& filePath,
													  bool generateMips) {
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
		texture = createTexture(device, commandPool, pixels, size, extent, VK_FORMAT_R8G8B8A8_SRGB, generateMips);
	}
	catch (...) {
		stbi_image_free(pixels);
		throw;
	}

	stbi_image_free(pixels);
	return texture;
}

void foto::Image::transitionLayout(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout,
								   VkImageLayout newLayout, VkImageAspectFlags aspect, u32 baseMip, u32 mipCount) {
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
		.subresourceRange = {aspect, baseMip, mipCount, 0, VK_REMAINING_ARRAY_LAYERS},
	};

	VkDependencyInfo dependency = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier,
	};
	vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

foto::u32 foto::Image::fullMipCount(VkExtent2D extent) {
	u32 largest = std::max(extent.width, extent.height);
	u32 levels = 1;
	while (largest > 1) {
		largest /= 2;
		levels++;
	}
	return levels;
}

void foto::Image::generateMipmaps(VkCommandBuffer commandBuffer, VkImage image, VkExtent2D extent, u32 mipLevels) {
	i32 srcWidth = static_cast<i32>(extent.width);
	i32 srcHeight = static_cast<i32>(extent.height);

	for (u32 level = 1; level < mipLevels; level++) {
		i32 dstWidth = std::max(srcWidth / 2, 1);
		i32 dstHeight = std::max(srcHeight / 2, 1);

		// Previous level is finished being written: make it the blit source
		transitionLayout(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						 VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1);

		// Downscale previous level (filtered) into this one
		VkImageBlit blit = {
			.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1},
			.srcOffsets = {{0, 0, 0}, {srcWidth, srcHeight, 1}},
			.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
			.dstOffsets = {{0, 0, 0}, {dstWidth, dstHeight, 1}},
		};
		vkCmdBlitImage(commandBuffer,
					   image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
					   image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					   1, &blit, VK_FILTER_LINEAR);

		// Previous level is done being read: ready to be sampled
		transitionLayout(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
						 VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1);

		srcWidth = dstWidth;
		srcHeight = dstHeight;
	}

	// Last level was only ever written to, never used as a source
	transitionLayout(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					 VK_IMAGE_ASPECT_COLOR_BIT, mipLevels - 1, 1);
}

VkImage foto::Image::createImage(Device& device, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage, u32 mipLevels) {
	VkImageCreateInfo imageCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = {extent.width, extent.height, 1},
		.mipLevels = mipLevels,
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

VkImageView foto::Image::createImageView(Device& device, VkImage image, VkFormat format, VkImageAspectFlags aspect, u32 mipLevels) {
	VkImageViewCreateInfo viewCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {aspect, 0, mipLevels, 0, 1}
	};

	VkImageView view = VK_NULL_HANDLE;

	if (vkCreateImageView(device.getDevice(), &viewCreateInfo, nullptr, &view) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkImage.cpp -> createImageView()");
	}

	return view;
}


foto::Sampler::Sampler(Device& device, VkFilter filter, VkSamplerAddressMode addressMode, f32 maxAnisotropy) :
	_device(device.getDevice()), sampler(createSampler(device, filter, addressMode, maxAnisotropy))
{

};

foto::Sampler::~Sampler() {
	vkDestroySampler(_device, sampler, nullptr);
};

VkSampler foto::Sampler::createSampler(Device& device, VkFilter filter, VkSamplerAddressMode addressMode, f32 maxAnisotropy) {
	// Clamp to what the GPU allows (1.0 = feature unsupported/disabled)
	f32 anisotropy = std::min(maxAnisotropy, device.getMaxSamplerAnisotropy());

	VkSamplerCreateInfo samplerCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = filter,
		.minFilter = filter,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR, // blend between mip levels (trilinear)
		.addressModeU = addressMode,
		.addressModeV = addressMode,
		.addressModeW = addressMode,
		.mipLodBias = 0.0f,
		.anisotropyEnable = anisotropy > 1.0f ? VK_TRUE : VK_FALSE,
		.maxAnisotropy = anisotropy,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE, // use every mip level the image has
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE
	};

	VkSampler sampler = VK_NULL_HANDLE;

	if (vkCreateSampler(device.getDevice(), &samplerCreateInfo, nullptr, &sampler) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkImage.cpp -> createSampler()");
	}

	return sampler;
}
