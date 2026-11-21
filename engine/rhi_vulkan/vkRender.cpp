#include "vkRender.hpp"
#include "vkImage.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

foto::Render::Render(Device& device, Swapchain& swapchain, Pipeline& pipeline, Pipeline& presentPipeline, Pipeline& shadowPipeline) :
	device(device), swapchain(swapchain), pipeline(pipeline), presentPipeline(presentPipeline), shadowPipeline(shadowPipeline),
	_device(device.getDevice()), _commandPool(createCommandPool(device)),
	commandBuffers(createCommandBuffers(device, _commandPool)),
	uniformBuffers(createUniformBuffers(device)),
	// Pool sized from the pipeline's set 0 bindings, one set per frame in flight
	descriptorPool(device, MAX_FRAMES_IN_FLIGHT, pipeline.getDescriptorSetLayoutInfo(0).poolSizes(MAX_FRAMES_IN_FLIGHT)),
	descriptorSets(descriptorPool.allocate(pipeline.getDescriptorSetLayout(0), MAX_FRAMES_IN_FLIGHT)),
	materialBuffers(createMaterialBuffers(device)),
	depthImage(createDepthImage(device, swapchain.getExtent()))
{
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(device.getPhysicalDevice(), &properties);
	if (properties.limits.maxPerStageDescriptorSampledImages < MAX_TEXTURES ||
		properties.limits.maxPerStageDescriptorSamplers < MAX_TEXTURES) {
		throw std::runtime_error("ERROR: vkRender.cpp -> Render(): GPU can't bind MAX_TEXTURES textures");
	}

	for (size i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		DescriptorWriter()
			.writeBuffer(0, uniformBuffers[i]->getBuffer(), sizeof(FrameUbo))
			.writeBuffer(1, materialBuffers[i]->getBuffer(), sizeof(Material) * MAX_MATERIALS, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
			.update(device, descriptorSets[i]);
	}

	const u8 white[4] = { 255, 255, 255, 255 };
	defaultTexture = Image::createTexture(device, _commandPool, white, sizeof(white), { 1, 1 }, VK_FORMAT_R8G8B8A8_SRGB, false);
	defaultSampler = std::make_unique<Sampler>(device);
	addTexture(*defaultTexture, *defaultSampler);
	addMaterial(Material{});

	hdrSampler = std::make_unique<Sampler>(device, VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	hdrImage = createHdrImage(device, swapchain.getExtent());
	hdrTextureSlot = addTexture(*hdrImage, *hdrSampler);
	tonemapMaterial = addMaterial({ .baseColorTexture = hdrTextureSlot });

	staticShadowAtlas = createShadowAtlas(device);
	dynamicShadowAtlas = createShadowAtlas(device);
	staticShadowSlot = addTexture(*staticShadowAtlas, *hdrSampler);
	dynamicShadowSlot = addTexture(*dynamicShadowAtlas, *hdrSampler);

	createSyncObjects(_device, swapchain.getImages().size());
};

foto::Render::~Render() {
	for (VkSemaphore sem : imageAvailableSem) vkDestroySemaphore(_device, sem, nullptr);
	for (VkSemaphore sem : renderFinishedSem) vkDestroySemaphore(_device, sem, nullptr);
	for (VkFence fence : inFlightFence) vkDestroyFence(_device, fence, nullptr);

	vkDestroyCommandPool(_device, _commandPool, nullptr);
	// descriptorPool (and its sets) + uniformBuffers are destroyed by their own destructors after this
};

foto::u32 foto::Render::addTexture(Image& texture, Sampler& sampler) {
	if (textureCount >= MAX_TEXTURES) {
		throw std::runtime_error("ERROR: vkRender.cpp -> addTexture(): MAX_TEXTURES reached");
	}

	vkDeviceWaitIdle(_device);
	writeTextureSlot(textureCount, texture, sampler);
	return textureCount++;
}

void foto::Render::writeTextureSlot(u32 slot, Image& texture, Sampler& sampler) {
	DescriptorWriter writer;
	writer.writeImage(2, texture.getView(), sampler.getSampler(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, slot);
	for (VkDescriptorSet set : descriptorSets) {
		writer.update(device, set);
	}
}

foto::u32 foto::Render::addMaterial(const Material& material) {
	if (materials.size() >= MAX_MATERIALS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> addMaterial(): MAX_MATERIALS reached");
	}

	materials.push_back(material);
	materialsVersion++;
	return static_cast<u32>(materials.size() - 1);
}

void foto::Render::updateMaterial(u32 index, const Material& material) {
	materials.at(index) = material;
	materialsVersion++;
}

// CPU -> recording command buffer
// GPU -> submitting/executing command buffer
bool foto::Render::drawFrame(std::span<const DrawItem> items, const FrameUbo& frameUbo,
							 const std::function<void(VkCommandBuffer)>& overlay) {
	vkWaitForFences(device.getDevice(), 1, &inFlightFence[currentFrame], VK_TRUE, UINT64_MAX);

	u32 imageIdx;
	VkResult acquireResult = vkAcquireNextImageKHR(device.getDevice(), swapchain.getSwapchain(), UINT64_MAX, imageAvailableSem[currentFrame], VK_NULL_HANDLE, &imageIdx);
	if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
		return true;
	}
	if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("ERROR: vkRender.cpp -> drawFrame() acquire");
	}

	// Reset only once we know we will submit, otherwise an early return leaves it unsignaled -> deadlock
	vkResetFences(device.getDevice(), 1, &inFlightFence[currentFrame]);

	uniformBuffers[currentFrame]->upload(&frameUbo, sizeof(frameUbo));

	if (uploadedMaterialsVersion[currentFrame] != materialsVersion) {
		materialBuffers[currentFrame]->upload(materials.data(), sizeof(Material) * materials.size());
		uploadedMaterialsVersion[currentFrame] = materialsVersion;
	}

	VkCommandBuffer commandBuffer = commandBuffers[currentFrame];
	vkResetCommandBuffer(commandBuffer, 0);
	bool capture = prepareScreenshot();
	recordCommandBuffer(commandBuffer, imageIdx, items, frameUbo, overlay, capture);

	// Submit info
	VkSemaphore waitSemaphore[] = { imageAvailableSem[currentFrame] };
	VkPipelineStageFlags waitStage[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkSemaphore signalSemaphore[] = { renderFinishedSem[imageIdx] };

	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = waitSemaphore,
		.pWaitDstStageMask = waitStage,

		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer,

		.signalSemaphoreCount = 1,
		.pSignalSemaphores = signalSemaphore,
	};

	// Fence must be unsignaled, same with signalSemaphores
	if (vkQueueSubmit(device.getGraphicsQ(), 1, &submitInfo, inFlightFence[currentFrame]) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> drawFrame()");
	}

	if (capture) {
		vkQueueWaitIdle(device.getGraphicsQ());
		saveScreenshot();
	}

	// Present queue
	VkSwapchainKHR swapchains[] = { swapchain.getSwapchain()};

	VkPresentInfoKHR presentInfo = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,

		.waitSemaphoreCount = 1,
		.pWaitSemaphores = signalSemaphore,

		.swapchainCount = 1,
		.pSwapchains = swapchains,

		.pImageIndices = &imageIdx,
	};

	// vkQueuePresentKHR doesn't wait, returns right away
	VkResult presentResult = vkQueuePresentKHR(device.getPresentQ(), &presentInfo);

	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

	if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
		return true;
	}
	if (presentResult != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> drawFrame() present");
	}
	return false;
}

void foto::Render::onSwapchainRecreated() {
	for (VkSemaphore sem : renderFinishedSem) vkDestroySemaphore(_device, sem, nullptr);
	renderFinishedSem.clear();
	createRenderFinishedSems(_device, swapchain.getImages().size());

	// Swapchain::recreate() already waited for the device to go idle, so the old one isn't in use
	depthImage = createDepthImage(device, swapchain.getExtent());
	hdrImage = createHdrImage(device, swapchain.getExtent());
	writeTextureSlot(hdrTextureSlot, *hdrImage, *hdrSampler);
}

void foto::Render::createSyncObjects(VkDevice device, size swapchainImageCount) {
	VkSemaphoreCreateInfo semaphoreCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};

	VkFenceCreateInfo fenceCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	// Per frame in flight: CPU frame pacing
	imageAvailableSem.resize(MAX_FRAMES_IN_FLIGHT);
	inFlightFence.resize(MAX_FRAMES_IN_FLIGHT);

	for (size i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &imageAvailableSem[i]) != VK_SUCCESS ||
			vkCreateFence(device, &fenceCreateInfo, nullptr, &inFlightFence[i]) != VK_SUCCESS) {
			throw std::runtime_error("ERROR: vkRender.cpp -> createSyncObjects()");
		}
	}

	createRenderFinishedSems(device, swapchainImageCount);
}

void foto::Render::createRenderFinishedSems(VkDevice device, size swapchainImageCount) {
	VkSemaphoreCreateInfo semaphoreCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};

	// Per swapchain image: presentation engine holds it until that image is reacquired
	renderFinishedSem.resize(swapchainImageCount);

	for (size i = 0; i < swapchainImageCount; i++) {
		if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderFinishedSem[i]) != VK_SUCCESS) {
			throw std::runtime_error("ERROR: vkRender.cpp -> createSyncObjects()");
		}
	}
}

void foto::Render::recordCommandBuffer(VkCommandBuffer commandBuffer, u32 imageIdx, std::span<const DrawItem> items,
									   const FrameUbo& frameUbo, const std::function<void(VkCommandBuffer)>& overlay, bool capture) {
	VkCommandBufferBeginInfo beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	};

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> recordCommandBuffer()");
	}

	VkExtent2D extent = swapchain.getExtent();
	VkImage swapchainImage = swapchain.getImages()[imageIdx];

	recordShadowPass(commandBuffer, *staticShadowAtlas, ShadowCaster::Static, items, frameUbo);
	recordShadowPass(commandBuffer, *dynamicShadowAtlas, ShadowCaster::Dynamic, items, frameUbo);

	VkImageMemoryBarrier2 beginBarriers[] = {
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = hdrImage->getImage(),
			.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
		},
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
			.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
			.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = depthImage->getImage(),
			.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
		},
		{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			.srcAccessMask = 0,
			.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
			.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = swapchainImage,
			.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
		},
	};

	VkDependencyInfo beginDependency = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 3,
		.pImageMemoryBarriers = beginBarriers,
	};
	vkCmdPipelineBarrier2(commandBuffer, &beginDependency);

	VkRenderingAttachmentInfo hdrAttachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = hdrImage->getView(),
		.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = {
			.color = {
				.float32 = {0.1f, 0.1f, 0.1f, 1.0f}
			}
		}
	};

	VkRenderingAttachmentInfo depthAttachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = depthImage->getView(),
		.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.clearValue = {
			.depthStencil = {1.0f, 0}
		}
	};

	VkRenderingInfo sceneRendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { .offset = {0, 0}, .extent = extent },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &hdrAttachment,
		.pDepthAttachment = &depthAttachment,
	};
	vkCmdBeginRendering(commandBuffer, &sceneRendering);
	setSceneViewport(commandBuffer, extent);

	Pipeline* boundPipeline = nullptr;
	Mesh* boundMesh = nullptr;
	for (const DrawItem& item : items) {
		Pipeline* itemPipeline = item.pipeline ? item.pipeline : &pipeline;
		if (!item.mesh && !item.pipeline) continue;

		if (itemPipeline != boundPipeline) {
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, itemPipeline->getPipeline());
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, itemPipeline->getPipelineLayout(),
									0, 1, &descriptorSets[currentFrame], 0, nullptr);
			boundPipeline = itemPipeline;
		}

		if (!item.mesh) {
			DrawPush push = { .model = item.transform, .materialIndex = (item.materialOverride != NO_MATERIAL_OVERRIDE) ? item.materialOverride : DEFAULT_MATERIAL };
			vkCmdPushConstants(commandBuffer, itemPipeline->getPipelineLayout(), DRAW_PUSH_RANGE.stageFlags, 0, sizeof(DrawPush), &push);
			vkCmdDraw(commandBuffer, 3, 1, 0, 0);
			continue;
		}

		if (item.mesh != boundMesh) {
			item.mesh->bind(commandBuffer);
			boundMesh = item.mesh;
		}

		for (const Submesh& submesh : item.mesh->getSubmeshes()) {
			DrawPush push = {
				.model = item.transform,
				.materialIndex = (item.materialOverride != NO_MATERIAL_OVERRIDE) ? item.materialOverride : submesh.materialIndex
			};
			vkCmdPushConstants(commandBuffer, itemPipeline->getPipelineLayout(), DRAW_PUSH_RANGE.stageFlags, 0, sizeof(DrawPush), &push);
			vkCmdDrawIndexed(commandBuffer, submesh.indexCount, 1, submesh.firstIndex, 0, 0);
		}
	}

	vkCmdEndRendering(commandBuffer);

	VkImageMemoryBarrier2 hdrToSampled = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = hdrImage->getImage(),
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};

	VkDependencyInfo hdrDependency = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &hdrToSampled,
	};
	vkCmdPipelineBarrier2(commandBuffer, &hdrDependency);

	VkRenderingAttachmentInfo swapchainAttachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = swapchain.getImageViews()[imageIdx],
		.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	};

	VkRenderingInfo presentRendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { .offset = {0, 0}, .extent = extent },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &swapchainAttachment,
	};
	vkCmdBeginRendering(commandBuffer, &presentRendering);
	setFullViewport(commandBuffer, extent);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipeline.getPipeline());
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipeline.getPipelineLayout(),
							0, 1, &descriptorSets[currentFrame], 0, nullptr);
	DrawPush tonemapPush = { .model = enginemath::Mat4::identity(), .materialIndex = tonemapMaterial };
	vkCmdPushConstants(commandBuffer, presentPipeline.getPipelineLayout(), DRAW_PUSH_RANGE.stageFlags, 0, sizeof(DrawPush), &tonemapPush);
	vkCmdDraw(commandBuffer, 3, 1, 0, 0);

	if (overlay) overlay(commandBuffer);

	vkCmdEndRendering(commandBuffer);

	VkImageLayout finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	if (capture) {
		Image::transitionLayout(commandBuffer, swapchainImage,
								VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

		VkBufferImageCopy copyRegion = {
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
			.imageOffset = {0, 0, 0},
			.imageExtent = {extent.width, extent.height, 1}
		};
		vkCmdCopyImageToBuffer(commandBuffer, swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
							   screenshotBuffer->getBuffer(), 1, &copyRegion);
		finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	}

	VkImageMemoryBarrier2 toPresent = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = capture ? VK_PIPELINE_STAGE_2_COPY_BIT : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = capture ? VK_ACCESS_2_NONE : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_NONE,
		.oldLayout = finalLayout,
		.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = swapchainImage,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};

	VkDependencyInfo toPresentDep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &toPresent,
	};
	vkCmdPipelineBarrier2(commandBuffer, &toPresentDep);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> recordCommandBuffer()");
	}
}

VkRect2D foto::Render::getSceneRect() {
	VkExtent2D extent = swapchain.getExtent();
	if (viewportAspect <= 0.0f || extent.height == 0) return { {0, 0}, extent };

	f32 windowAspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);
	if (windowAspect > viewportAspect) {
		u32 width = static_cast<u32>(extent.height * viewportAspect + 0.5f);
		return { { static_cast<i32>((extent.width - width) / 2), 0 }, { width, extent.height } };
	}
	u32 height = static_cast<u32>(extent.width / viewportAspect + 0.5f);
	return { { 0, static_cast<i32>((extent.height - height) / 2) }, { extent.width, height } };
}

void foto::Render::setSceneViewport(VkCommandBuffer commandBuffer, VkExtent2D extent) {
	VkRect2D rect = getSceneRect();
	VkViewport viewport = {
		.x = static_cast<f32>(rect.offset.x),
		.y = static_cast<f32>(rect.offset.y),
		.width = static_cast<f32>(rect.extent.width),
		.height = static_cast<f32>(rect.extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
	vkCmdSetScissor(commandBuffer, 0, 1, &rect);
}

void foto::Render::recordShadowPass(VkCommandBuffer commandBuffer, Image& atlas, ShadowCaster group,
									std::span<const DrawItem> items, const FrameUbo& frameUbo) {
	VkImageMemoryBarrier2 toDepth = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		.srcAccessMask = VK_ACCESS_2_NONE,
		.dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
		.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = atlas.getImage(),
		.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
	};
	VkDependencyInfo toDepthDependency = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toDepth };
	vkCmdPipelineBarrier2(commandBuffer, &toDepthDependency);

	VkRenderingAttachmentInfo depthAttachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = atlas.getView(),
		.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = { .depthStencil = {1.0f, 0} }
	};
	VkRenderingInfo rendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { .offset = {0, 0}, .extent = atlas.getExtent() },
		.layerCount = 1,
		.pDepthAttachment = &depthAttachment,
	};
	vkCmdBeginRendering(commandBuffer, &rendering);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline.getPipeline());
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline.getPipelineLayout(),
							0, 1, &descriptorSets[currentFrame], 0, nullptr);

	u32 lightCount = std::min(frameUbo.lightInfo[0], MAX_LIGHTS);
	for (u32 light = 0; light < lightCount; light++) {
		if (frameUbo.lights[light].position.w < 0.5f) continue;

		for (u32 face = 0; face < 6; face++) {
			VkViewport viewport = {
				.x = static_cast<f32>(face * SHADOW_FACE_RESOLUTION),
				.y = static_cast<f32>(light * SHADOW_FACE_RESOLUTION),
				.width = static_cast<f32>(SHADOW_FACE_RESOLUTION),
				.height = static_cast<f32>(SHADOW_FACE_RESOLUTION),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};
			VkRect2D scissor = {
				.offset = { static_cast<i32>(face * SHADOW_FACE_RESOLUTION), static_cast<i32>(light * SHADOW_FACE_RESOLUTION) },
				.extent = { SHADOW_FACE_RESOLUTION, SHADOW_FACE_RESOLUTION }
			};
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			Mesh* boundMesh = nullptr;
			for (const DrawItem& item : items) {
				if (!item.mesh || item.shadowCaster != group) continue;
				if (item.mesh != boundMesh) {
					item.mesh->bind(commandBuffer);
					boundMesh = item.mesh;
				}
				DrawPush push = { .model = item.transform, .materialIndex = light * 6 + face };
				vkCmdPushConstants(commandBuffer, shadowPipeline.getPipelineLayout(), DRAW_PUSH_RANGE.stageFlags, 0, sizeof(DrawPush), &push);
				for (const Submesh& submesh : item.mesh->getSubmeshes()) {
					vkCmdDrawIndexed(commandBuffer, submesh.indexCount, 1, submesh.firstIndex, 0, 0);
				}
			}
		}
	}

	vkCmdEndRendering(commandBuffer);

	VkImageMemoryBarrier2 toSampled = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
		.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = atlas.getImage(),
		.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
	};
	VkDependencyInfo toSampledDependency = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toSampled };
	vkCmdPipelineBarrier2(commandBuffer, &toSampledDependency);
}

void foto::Render::setFullViewport(VkCommandBuffer commandBuffer, VkExtent2D extent) {
	VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<f32>(extent.width),
		.height = static_cast<f32>(extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor = { .offset = {0, 0}, .extent = extent };
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

VkCommandPool foto::Render::createCommandPool(Device& device) {
	std::optional<QueueInfos> QFamilyIdx = device.findQueueFamilies(device.getPhysicalDevice(), device.getSurface());

	VkCommandPoolCreateInfo poolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = QFamilyIdx.value().graphicsQueueIdx.value()
	};

	VkCommandPool commandPool = VK_NULL_HANDLE;

	if (vkCreateCommandPool(device.getDevice(), &poolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> createCommandPool()");
	}

	return commandPool;
}

std::vector<VkCommandBuffer> foto::Render::createCommandBuffers(Device& device, VkCommandPool commandPool) {
	std::vector<VkCommandBuffer> commandBuffers(MAX_FRAMES_IN_FLIGHT);

	VkCommandBufferAllocateInfo commandBufferAllocInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = commandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = static_cast<u32>(commandBuffers.size())
	};

	if (vkAllocateCommandBuffers(device.getDevice(), &commandBufferAllocInfo, commandBuffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> createCommandBuffers()");
	}

	return commandBuffers;
}

std::vector<std::unique_ptr<foto::Buffer>> foto::Render::createUniformBuffers(Device& device) {
	std::vector<std::unique_ptr<Buffer>> buffers;
	buffers.reserve(MAX_FRAMES_IN_FLIGHT);

	// HOST_VISIBLE: rewritten from the CPU every frame, so no staging buffer
	for (size i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		buffers.push_back(std::make_unique<Buffer>(device, sizeof(FrameUbo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
	}

	return buffers;
}

std::vector<std::unique_ptr<foto::Buffer>> foto::Render::createMaterialBuffers(Device& device) {
	std::vector<std::unique_ptr<Buffer>> buffers;
	buffers.reserve(MAX_FRAMES_IN_FLIGHT);

	for (size i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		buffers.push_back(std::make_unique<Buffer>(device, sizeof(Material) * MAX_MATERIALS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
	}

	return buffers;
}

bool foto::Render::prepareScreenshot() {
	if (screenshotPath.empty()) return false;

	SwapchainSupportDetails support = Device::querySwapchainSupport(device.getPhysicalDevice(), device.getSurface());
	if (!(support.capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
		std::cerr << "[foto] screenshot failed: swapchain images can't be copied on this device\n";
		screenshotPath.clear();
		return false;
	}

	VkDeviceSize size = static_cast<VkDeviceSize>(swapchain.getExtent().width) * swapchain.getExtent().height * 4;
	if (!screenshotBuffer || screenshotBuffer->getSize() != size) {
		screenshotBuffer = std::make_unique<Buffer>(device, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	}
	return true;
}

void foto::Render::saveScreenshot() {
	VkExtent2D extent = swapchain.getExtent();
	std::vector<u8> pixels(screenshotBuffer->getSize());
	screenshotBuffer->download(pixels.data(), pixels.size());

	VkFormat format = swapchain.getFormat();
	bool bgra = format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_B8G8R8A8_UNORM;
	for (size i = 0; i < pixels.size(); i += 4) {
		if (bgra) std::swap(pixels[i], pixels[i + 2]);
		pixels[i + 3] = 255;
	}

	if (stbi_write_png(screenshotPath.c_str(), static_cast<int>(extent.width), static_cast<int>(extent.height), 4,
					   pixels.data(), static_cast<int>(extent.width) * 4)) {
		std::cout << "[foto] saved screenshot " << screenshotPath << std::endl;
	}
	else {
		std::cerr << "[foto] screenshot failed: couldn't write " << screenshotPath << "\n";
	}
	screenshotPath.clear();
}

std::unique_ptr<foto::Image> foto::Render::createHdrImage(Device& device, VkExtent2D extent) {
	return std::make_unique<Image>(device, extent, HDR_FORMAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
}

std::unique_ptr<foto::Image> foto::Render::createShadowAtlas(Device& device) {
	return std::make_unique<Image>(device, VkExtent2D{ 6 * SHADOW_FACE_RESOLUTION, MAX_LIGHTS * SHADOW_FACE_RESOLUTION }, DEPTH_FORMAT,
								   VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void foto::shadowFaceMatrices(const enginemath::Vec3& position, enginemath::Mat4 out[6]) {
	using enginemath::Vec3;
	using enginemath::Mat4;
	const Vec3 directions[6] = { Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0), Vec3(0, 0, 1), Vec3(0, 0, -1) };
	const Vec3 ups[6] = { Vec3(0, 1, 0), Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(0, 0, 1), Vec3(0, 1, 0), Vec3(0, 1, 0) };

	Mat4 projection = Mat4::projectionM(enginemath::toRad(90.0f), 1.0f, 0.02f, 50.0f);
	for (int face = 0; face < 6; face++) {
		out[face] = projection * Mat4::lookAtM(position, position + directions[face], ups[face]);
	}
}

std::unique_ptr<foto::Image> foto::Render::createDepthImage(Device& device, VkExtent2D extent) {
	return std::make_unique<Image>(device, extent, DEPTH_FORMAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
								   VK_IMAGE_ASPECT_DEPTH_BIT);
}