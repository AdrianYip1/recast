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
#include "vkAccel.hpp"

namespace foto {
	class Image;
	class Sampler;

	class Render {
	public:
		// Depth attachment format. Pipelines drawn by Render need it as GraphicsPipelineConfig::depthFormat
		static constexpr VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;
		// Scene pipelines render into this; presentPipeline tonemaps it to the swapchain
		static constexpr VkFormat HDR_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;

		// pathTracePipeline / causticsPipeline: nullptr when the GPU can't ray trace
		Render(Device& device, Swapchain& swapchain, Pipeline& pipeline, Pipeline& presentPipeline, Pipeline& shadowPipeline,
			   Pipeline* pathTracePipeline, Pipeline* causticsPipeline);
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

		// Letterboxes the scene pass to this width/height ratio (0 = fill the window)
		void setViewportAspect(f32 aspect) { viewportAspect = aspect; };
		VkRect2D getSceneRect();

		// Bindless texture slots of the static and dynamic shadow atlases (for FrameUbo::lightInfo)
		u32 getStaticShadowAtlas() { return staticShadowSlot; };
		u32 getDynamicShadowAtlas() { return dynamicShadowSlot; };

		// Replaces the shadow + scene passes with pathtrace.comp, which composites the dynamic casters into photoTexture.
		// Samples accumulate over frames and restart whenever the casters, lights or camera change
		void setPathTracing(bool enabled, u32 photoTexture);
		bool isPathTracing() { return pathTracing; };
		bool supportsPathTracing() { return pathTracePipeline != nullptr; };
		u32 getPathTraceSamples() { return traceSamples; };
		u32 getPathTracePhoto() { return tracePhotoTexture; };

		VkCommandPool getCommandPool() { return _commandPool; };
	private:
		Device& device;
		Swapchain& swapchain;
		Pipeline& pipeline;
		Pipeline& presentPipeline;
		Pipeline& shadowPipeline;
		Pipeline* pathTracePipeline = nullptr;
		Pipeline* causticsPipeline = nullptr;

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

		std::unique_ptr<Image> hdrImage;
		std::unique_ptr<Sampler> hdrSampler;
		u32 hdrTextureSlot = 0;
		u32 tonemapMaterial = 0;
		f32 viewportAspect = 0.0f;

		std::unique_ptr<Image> staticShadowAtlas;
		std::unique_ptr<Image> dynamicShadowAtlas;
		u32 staticShadowSlot = 0;
		u32 dynamicShadowSlot = 0;

		bool pathTracing = false;
		u32 tracePhotoTexture = 0;
		u32 traceSamples = 0;
		std::vector<u8> traceKey;
		std::unique_ptr<DescriptorPool> traceDescriptorPool;
		std::vector<VkDescriptorSet> traceSets; // set 1 of pathtrace.comp, one per frame in flight
		std::vector<std::unique_ptr<TopLevelAccelerationStructure>> topLevels;
		std::vector<std::unique_ptr<Buffer>> geometryBuffers;
		std::unique_ptr<Image> accumulationImage;
		std::unique_ptr<Image> causticImage; // R32_UINT, 3 columns (rgb) per framebuffer pixel
		std::unique_ptr<Image> illuminationImage;
		std::unique_ptr<Image> guideImage;

		std::vector<VkSemaphore> imageAvailableSem;
		std::vector<VkSemaphore> renderFinishedSem;
		std::vector<VkFence> inFlightFence;

		u32 currentFrame = 0;

		void createSyncObjects(VkDevice device, size swapchainImageCount);
		void createRenderFinishedSems(VkDevice device, size swapchainImageCount);
		void recordCommandBuffer(VkCommandBuffer commandBuffer, u32 imageIdx, std::span<const DrawItem> items,
								 const FrameUbo& frameUbo, const std::function<void(VkCommandBuffer)>& overlay, bool capture);
		void setFullViewport(VkCommandBuffer commandBuffer, VkExtent2D extent);
		void recordRasterScene(VkCommandBuffer commandBuffer, std::span<const DrawItem> items, const FrameUbo& frameUbo, VkExtent2D extent);
		void recordPathTrace(VkCommandBuffer commandBuffer, std::span<const DrawItem> items, const FrameUbo& frameUbo, VkExtent2D extent);
		void updateTraceSamples(std::span<const DrawItem> items, const FrameUbo& frameUbo);
		void createTraceImages();
		void recordShadowPass(VkCommandBuffer commandBuffer, Image& atlas, ShadowCaster group, std::span<const DrawItem> items, const FrameUbo& frameUbo);
		void setSceneViewport(VkCommandBuffer commandBuffer, VkExtent2D extent);
		void writeTextureSlot(u32 slot, Image& texture, Sampler& sampler);
		bool prepareScreenshot();
		void saveScreenshot();
		static VkCommandPool createCommandPool(Device& device);
		static std::vector<VkCommandBuffer> createCommandBuffers(Device& device, VkCommandPool commandPool);
		static std::vector<std::unique_ptr<Buffer>> createUniformBuffers(Device& device);
		static std::vector<std::unique_ptr<Buffer>> createMaterialBuffers(Device& device);
		static std::unique_ptr<Image> createDepthImage(Device& device, VkExtent2D extent);
		static std::unique_ptr<Image> createHdrImage(Device& device, VkExtent2D extent);
		static std::unique_ptr<Image> createShadowAtlas(Device& device);
	};
} // namespace foto