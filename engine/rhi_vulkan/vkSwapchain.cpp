#include "vkSwapchain.hpp"

foto::Swapchain::Swapchain(GLFWwindow* window, Device& device) :
	swapchainSupport(device.querySwapchainSupport(device.getPhysicalDevice(), device.getSurface())),
	swapchain(createSwapchain(window, device, swapchainSupport)),
	device(device.getDevice())
{

}

foto::Swapchain::~Swapchain() {
	vkDestroySwapchainKHR(device, swapchain, nullptr);
}

VkSwapchainKHR foto::Swapchain::createSwapchain(GLFWwindow* window, Device& device, SwapchainSupportDetails swapchainSupport) {
	VkSwapchainKHR swapchain = VK_NULL_HANDLE;

	// Queue
	QueueInfos qInfo = device.findQueueFamilies(device.getPhysicalDevice(), device.getSurface()).value();
	u32 queueFamilyIndices[] = { qInfo.graphicsQueueIdx.value(), qInfo.presentQueueIdx.value()};
	bool isConcurrent = (qInfo.graphicsQueueIdx.value() != qInfo.presentQueueIdx.value());

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

	// Find format
	VkSurfaceFormatKHR format{};
	bool foundFormat = false;

	for (const auto& availableFormat : swapchainSupport.formats) {
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
			availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			format = availableFormat;
			foundFormat = true;
			break;
		}
	}
	if (!foundFormat) format = swapchainSupport.formats[0];

	// Find present mode
	VkPresentModeKHR presentMode{};
	bool foundPresentMode = false;

	for (const auto& availablePresentMode : swapchainSupport.presentModes) {
		if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
			presentMode = availablePresentMode;
			foundPresentMode = true;
			break;
		}
	}
	if (!foundPresentMode) presentMode = VK_PRESENT_MODE_FIFO_KHR;


	VkSwapchainCreateInfoKHR createInfo = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.surface = device.getSurface(),
		.minImageCount = minImageCount,
		.imageFormat = format.format,
		.imageColorSpace = format.colorSpace,
		.imageExtent = extent,
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.imageSharingMode = (isConcurrent) ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
		.queueFamilyIndexCount = (isConcurrent) ? static_cast<u32>(2) : static_cast<u32>(0),
		.pQueueFamilyIndices = (isConcurrent) ? queueFamilyIndices : nullptr,
		.preTransform = swapchainSupport.capabilities.currentTransform,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
		.presentMode = presentMode,
		.clipped = VK_TRUE,
	};

	if (vkCreateSwapchainKHR(device.getDevice(), &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkSwapchain.cpp -> createSwapchain()");
	};

	return swapchain;
}