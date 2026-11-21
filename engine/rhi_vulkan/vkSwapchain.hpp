#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"

namespace foto {

	class Swapchain {
	public:
		Swapchain(GLFWwindow* window, Device& device);
		~Swapchain();
		Swapchain(const Swapchain& other) = delete;
		Swapchain& operator=(const Swapchain& other) = delete;

		VkSwapchainKHR getSwapchain() { return swapchain; };
		VkFormat getFormat() { return _format; };
		VkExtent2D getExtent() { return _extent; };

		std::vector<VkImage> getImages() { return images; };
		std::vector<VkImageView> getImageViews() { return imageViews; };

	private:
		SwapchainSupportDetails swapchainSupport;
		VkFormat _format;
		VkExtent2D _extent;
		VkSwapchainKHR swapchain = VK_NULL_HANDLE;
		std::vector<VkImage> images;
		std::vector<VkImageView> imageViews;
		VkDevice device = VK_NULL_HANDLE;

		static VkSwapchainKHR createSwapchain(GLFWwindow* window, Device& device, SwapchainSupportDetails swapchainSupport,
											  VkFormat& _format, VkExtent2D& _extent);

		static std::vector<VkImage> getSwapchainImages(Device& device, VkSwapchainKHR swapchain);

		static std::vector<VkImageView> getImageViews(Device& device, std::vector<VkImage>& images, VkFormat _format);
	};
} // namespace foto