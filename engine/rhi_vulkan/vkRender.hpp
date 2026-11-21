#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"
#include "vkSwapchain.hpp"
#include "vkPipeline.hpp"

namespace foto {
	class Render {
	public:
		Render(Device& device, Swapchain& swapchain, Pipeline& pipeline);
		~Render();
		Render(const Render& other) = delete;
		Render& operator=(const Render& other) = delete;

	private:
		VkDevice device = VK_NULL_HANDLE;
		VkCommandPool _commandPool = VK_NULL_HANDLE;
		VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

		void recordCommandBuffer(Swapchain& swapchain, Pipeline& pipeline, VkCommandBuffer commandBuffer, u32 imageIdx);
		static VkCommandPool createCommandPool(Device& device);
		static VkCommandBuffer createCommandBuffer(Device& device, VkCommandPool commandPool);
	};
} // namespace foto