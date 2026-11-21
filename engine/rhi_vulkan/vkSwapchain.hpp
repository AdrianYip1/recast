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
	private:
		SwapchainSupportDetails swapchainSupport;

		static VkSwapchainKHR createSwapchain(GLFWwindow* window, Device& device, SwapchainSupportDetails swapchainSupport);
	};
} // namespace foto