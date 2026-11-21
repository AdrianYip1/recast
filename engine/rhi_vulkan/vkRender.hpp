#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"
#include "vkSwapchain.hpp"
#include "vkPipeline.hpp"

#include "vkBuffer.hpp"
#include "vkUbo.hpp"
#include "vkDescriptorHelper.hpp"

namespace foto {
	class Image;
	class Sampler;

	class Render {
	public:
		Render(Device& device, Swapchain& swapchain, Pipeline& pipeline);
		~Render();
		Render(const Render& other) = delete;
		Render& operator=(const Render& other) = delete;

		// Returns true if the swapchain is out of date / suboptimal and must be recreated
		bool drawFrame(const std::vector<std::unique_ptr<Buffer>>& vertexBuffers, Buffer& indexBuffer, const Ubo& ubo);

		// Call after Swapchain::recreate() (image count may change)
		void onSwapchainRecreated();

		// Points `binding` of every frame's descriptor set (set 0) at the texture.
		// Only call when no frame is in flight: at init, or after vkDeviceWaitIdle.
		void setTexture(u32 binding, Image& texture, Sampler& sampler);

		VkCommandPool getCommandPool() { return _commandPool; };
	private:
		Device& device;
		Swapchain& swapchain;
		Pipeline& pipeline;

		VkDevice _device = VK_NULL_HANDLE;
		VkCommandPool _commandPool = VK_NULL_HANDLE;
		std::vector<VkCommandBuffer> commandBuffers;

		std::vector<std::unique_ptr<Buffer>> uniformBuffers;
		DescriptorPool descriptorPool;
		std::vector<VkDescriptorSet> descriptorSets; // set 0, one per frame in flight, freed with the pool

		std::vector<VkSemaphore> imageAvailableSem;
		std::vector<VkSemaphore> renderFinishedSem;
		std::vector<VkFence> inFlightFence;

		u32 currentFrame = 0;

		void createSyncObjects(VkDevice device, size swapchainImageCount);
		void createRenderFinishedSems(VkDevice device, size swapchainImageCount);
		void recordCommandBuffer(VkCommandBuffer commandBuffer, u32 imageIdx,
								 const std::vector<std::unique_ptr<Buffer>>& vertexBuffers, Buffer& indexBuffer);
		static VkCommandPool createCommandPool(Device& device);
		static std::vector<VkCommandBuffer> createCommandBuffers(Device& device, VkCommandPool commandPool);
		static std::vector<std::unique_ptr<Buffer>> createUniformBuffers(Device& device);
	};
} // namespace foto