#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"
#include "vkSwapchain.hpp"
#include "vkPipeline.hpp"

#include "vkBuffer.hpp"
#include "vkMesh.hpp"
#include "vkRenderTypes.hpp"
#include "vkDescriptorHelper.hpp"

namespace foto {
	class Image;
	class Sampler;

	class Render {
	public:
		// Depth attachment format. Pipelines drawn by Render need it as GraphicsPipelineConfig::depthFormat
		static constexpr VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;

		Render(Device& device, Swapchain& swapchain, Pipeline& pipeline);
		~Render();
		Render(const Render& other) = delete;
		Render& operator=(const Render& other) = delete;

		// Returns true if the swapchain is out of date / suboptimal and must be recreated
		bool drawFrame(std::span<const DrawItem> items, const FrameUbo& frameUbo,
					   const std::function<void(VkCommandBuffer)>& overlay = nullptr);

		// Call after Swapchain::recreate() (image count may change)
		void onSwapchainRecreated();

		// Bindless texture slot for the shaders (Material::baseColorTexture). Waits for the GPU to go idle.
		u32 addTexture(Image& texture, Sampler& sampler);

		u32 addMaterial(const Material& material);
		void updateMaterial(u32 index, const Material& material);
		const Material& getMaterial(u32 index) { return materials.at(index); };

		// Saves the next presented frame (including the overlay) as a PNG
		void requestScreenshot(const std::string& filePath) { screenshotPath = filePath; };

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

		std::vector<Material> materials;
		std::vector<std::unique_ptr<Buffer>> materialBuffers;
		u64 materialsVersion = 1;
		std::array<u64, MAX_FRAMES_IN_FLIGHT> uploadedMaterialsVersion = {};

		u32 textureCount = 0;
		std::unique_ptr<Image> defaultTexture;
		std::unique_ptr<Sampler> defaultSampler;

		std::string screenshotPath;
		std::unique_ptr<Buffer> screenshotBuffer;

		// Swapchain-sized, shared by the frames in flight (the barrier at the start of each frame orders them)
		std::unique_ptr<Image> depthImage;

		std::vector<VkSemaphore> imageAvailableSem;
		std::vector<VkSemaphore> renderFinishedSem;
		std::vector<VkFence> inFlightFence;

		u32 currentFrame = 0;

		void createSyncObjects(VkDevice device, size swapchainImageCount);
		void createRenderFinishedSems(VkDevice device, size swapchainImageCount);
		void recordCommandBuffer(VkCommandBuffer commandBuffer, u32 imageIdx, std::span<const DrawItem> items,
								 const std::function<void(VkCommandBuffer)>& overlay, bool capture);
		bool prepareScreenshot();
		void saveScreenshot();
		static VkCommandPool createCommandPool(Device& device);
		static std::vector<VkCommandBuffer> createCommandBuffers(Device& device, VkCommandPool commandPool);
		static std::vector<std::unique_ptr<Buffer>> createUniformBuffers(Device& device);
		static std::vector<std::unique_ptr<Buffer>> createMaterialBuffers(Device& device);
		static std::unique_ptr<Image> createDepthImage(Device& device, VkExtent2D extent);
	};
} // namespace foto