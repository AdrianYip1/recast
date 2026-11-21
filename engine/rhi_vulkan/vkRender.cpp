#include "vkRender.hpp"
#include "vkImage.hpp"

foto::Render::Render(Device& device, Swapchain& swapchain, Pipeline& pipeline) :
	device(device), swapchain(swapchain), pipeline(pipeline),
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

	DescriptorWriter writer;
	writer.writeImage(2, texture.getView(), sampler.getSampler(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureCount);
	for (VkDescriptorSet set : descriptorSets) {
		writer.update(device, set);
	}

	return textureCount++;
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
bool foto::Render::drawFrame(std::span<const DrawItem> items, const FrameUbo& frameUbo) {
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
	recordCommandBuffer(commandBuffer, imageIdx, items);

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

void foto::Render::recordCommandBuffer(VkCommandBuffer commandBuffer, u32 imageIdx, std::span<const DrawItem> items) {
	VkCommandBufferBeginInfo beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	};

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> recordCommandBuffer()");
	}

	// Barrier 1 -> before rendering
	VkImageMemoryBarrier2 toColor = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = 0,
		.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,

		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = swapchain.getImages()[imageIdx],
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};

	// Depth: previous contents are discarded (cleared below). src = last frame's depth writes, so the
	// frames in flight sharing this one image never overlap
	VkImageMemoryBarrier2 toDepth = {
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
	};

	VkImageMemoryBarrier2 beginBarriers[] = { toColor, toDepth };

	VkDependencyInfo toColorDep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 2,
		.pImageMemoryBarriers = beginBarriers,
	};
	vkCmdPipelineBarrier2(commandBuffer, &toColorDep);

	// Begin rendering (instead of renderpass + framebuffer)
	VkRenderingAttachmentInfo colorAttachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = swapchain.getImageViews()[imageIdx],
		.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = {
			.color = {
				.float32 = {0.1f, 0.1f, 0.1f, 1.0f}
			}
		}
	};

	// Depth is only needed during the pass: cleared to the far plane, not stored
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

	VkRenderingInfo rendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = {
			.offset = {0, 0},
			.extent = swapchain.getExtent()
		},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachment,
		.pDepthAttachment = &depthAttachment,
	};
	vkCmdBeginRendering(commandBuffer, &rendering);

	// Bind pipeline/viewport/scissor
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getPipeline());

	// Bind descriptor
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getPipelineLayout(),
							0, 1, &descriptorSets[currentFrame], 0, nullptr);

	VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<f32>(swapchain.getExtent().width),
		.height = static_cast<f32>(swapchain.getExtent().height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

	VkRect2D scissor = {
		.offset = {0, 0},
		.extent = swapchain.getExtent(),
	};
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

	// Draw list: rebind buffers only when the mesh changes, one push + draw per submesh
	Mesh* boundMesh = nullptr;
	for (const DrawItem& item : items) {
		if (!item.mesh) continue;

		if (item.mesh != boundMesh) {
			item.mesh->bind(commandBuffer);
			boundMesh = item.mesh;
		}

		for (const Submesh& submesh : item.mesh->getSubmeshes()) {
			DrawPush push = {
				.model = item.transform,
				.materialIndex = (item.materialOverride != NO_MATERIAL_OVERRIDE) ? item.materialOverride : submesh.materialIndex
			};
			vkCmdPushConstants(commandBuffer, pipeline.getPipelineLayout(), DRAW_PUSH_RANGE.stageFlags, 0, sizeof(DrawPush), &push);
			vkCmdDrawIndexed(commandBuffer, submesh.indexCount, 1, submesh.firstIndex, 0, 0);
		}
	}

	vkCmdEndRendering(commandBuffer);

	// Barrier 2 -> after renderingg 
	VkImageMemoryBarrier2 toPresent = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_NONE,

		.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = swapchain.getImages()[imageIdx],
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

std::unique_ptr<foto::Image> foto::Render::createDepthImage(Device& device, VkExtent2D extent) {
	return std::make_unique<Image>(device, extent, DEPTH_FORMAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
								   VK_IMAGE_ASPECT_DEPTH_BIT);
}