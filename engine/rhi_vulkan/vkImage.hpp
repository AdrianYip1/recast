#pragma once

#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"

namespace foto {
	// Owns a VkImage, its memory and a view over the whole image (2D, all mip levels, 1 layer)
	class Image {
	public:
		Image(Device& device, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage,
			  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT, u32 mipLevels = 1);
		~Image();
		Image(const Image& other) = delete;
		Image& operator=(const Image& other) = delete;

		// GPU (DEVICE_LOCAL) sampled image filled with pixels via staging buffer, left in SHADER_READ_ONLY_OPTIMAL.
		// generateMips: builds the full mip chain on the GPU (falls back to 1 level if the format can't be linearly blitted)
		static std::unique_ptr<Image> createTexture(Device& device, VkCommandPool commandPool, const void* pixels,
													VkDeviceSize size, VkExtent2D extent,
													VkFormat format = VK_FORMAT_R8G8B8A8_SRGB, bool generateMips = true);

		// Loads a png/jpg/etc (via stb_image) as RGBA8 sRGB
		static std::unique_ptr<Image> loadTexture(Device& device, VkCommandPool commandPool, const std::string& filePath,
												  bool generateMips = true);

		// Records a layout transition of mip levels [baseMip, baseMip + mipCount).
		// Coarse (ALL_COMMANDS) sync: fine for load-time work, not for per-frame use
		static void transitionLayout(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout,
									 VkImageLayout newLayout, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
									 u32 baseMip = 0, u32 mipCount = VK_REMAINING_MIP_LEVELS);

		// Number of levels in a full chain down to 1x1: floor(log2(max(w, h))) + 1
		static u32 fullMipCount(VkExtent2D extent);

		VkImage getImage() { return image; };
		VkImageView getView() { return view; };
		VkFormat getFormat() { return _format; };
		VkExtent2D getExtent() { return _extent; };
		u32 getMipLevels() { return _mipLevels; };

	private:
		VkDevice _device = VK_NULL_HANDLE;
		VkExtent2D _extent = {};
		VkFormat _format = VK_FORMAT_UNDEFINED;
		u32 _mipLevels = 1;
		VkImage image = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkImageView view = VK_NULL_HANDLE;

		static VkImage createImage(Device& device, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage, u32 mipLevels);
		static VkDeviceMemory allocateMemory(Device& device, VkImage image);
		static VkImageView createImageView(Device& device, VkImage image, VkFormat format, VkImageAspectFlags aspect, u32 mipLevels);

		// Expects every level in TRANSFER_DST with level 0 filled. Leaves every level in SHADER_READ_ONLY
		static void generateMipmaps(VkCommandBuffer commandBuffer, VkImage image, VkExtent2D extent, u32 mipLevels);
	};

	// Owns a VkSampler. Independent of any image: one sampler can be used with many textures.
	// Trilinear (linear between mip levels) with anisotropic filtering when the GPU supports it.
	class Sampler {
	public:
		Sampler(Device& device, VkFilter filter = VK_FILTER_LINEAR,
				VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
				f32 maxAnisotropy = 16.0f);
		~Sampler();
		Sampler(const Sampler& other) = delete;
		Sampler& operator=(const Sampler& other) = delete;

		VkSampler getSampler() { return sampler; };

	private:
		VkDevice _device = VK_NULL_HANDLE;
		VkSampler sampler = VK_NULL_HANDLE;

		static VkSampler createSampler(Device& device, VkFilter filter, VkSamplerAddressMode addressMode, f32 maxAnisotropy);
	};
} // namespace foto
