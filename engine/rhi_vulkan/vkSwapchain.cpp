#include "vkSwapchain.hpp"

foto::Swapchain::Swapchain(GLFWwindow* window, Device& device) :
	swapchainSupport(device.querySwapchainSupport(device.getPhysicalDevice(), device.getSurface()))
{

}

foto::Swapchain::~Swapchain() {

}

VkSwapchainKHR foto::Swapchain::createSwapchain(GLFWwindow* window, Device& device, SwapchainSupportDetails swapchainSupport) {
	VkSwapchainKHR swapchain = VK_NULL_HANDLE;

	// Handle capabilities (maxImageCount = 0 means no limit)
	u32 minImageCount = swapchainSupport.capabilities.maxImageCount > 0 ? 
						std::min(swapchainSupport.capabilities.minImageCount + 1, swapchainSupport.capabilities.maxImageCount) :
						swapchainSupport.capabilities.minImageCount + 1;

	VkExtent2D extent = swapchainSupport.capabilities.currentExtent;
	if (extent.width == UINT32_MAX) {
		int framebufferW = 0, framebufferH = 0;
		glfwGetFramebufferSize(window, &framebufferW, &framebufferH);

		extent.width = std::clamp(static_cast<u32>(framebufferW), swapchainSupport.capabilities.minImageExtent.width, swapchainSupport.capabilities.maxImageExtent.width);
		extent.height = std::clamp(static_cast<u32>(framebufferH), swapchainSupport.capabilities.minImageExtent.height, swapchainSupport.capabilities.maxImageExtent.height);
	}


	VkSwapchainCreateInfoKHR createInfo = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.surface = device.getSurface(),
		.minImageCount = minImageCount,
		.imageFormat = ,
		.imageColorSpace = ,
		.imageExtent = extent,
		.imageArrayLayers = ,
		.imageUsage = ,
		.imageSharingMode ,
		.queueFamilyIndexCount = ,
		.pQueueFamilyIndices = ,
		.preTransform = ,
		.compositeAlpha = ,
		.presentMode = ,
		.clipped = ,
	};

	if (vkCreateSwapchainKHR(device.getDevice(), &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkSwapchain.cpp -> createSwapchain()");
	};
}