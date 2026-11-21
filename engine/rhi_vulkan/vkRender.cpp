#include "vkRender.hpp"

foto::Render::Render(Device& device, Swapchain& swapchain, Pipeline& pipeline) :
	_device(device.getDevice()), _commandPool(createCommandPool(device)), 
	commandBuffer(createCommandBuffer(device, _commandPool))
{
	createSyncObjects(_device);
};

foto::Render::~Render() {
	vkDestroySemaphore(_device, imageAvailableSem, nullptr);
	vkDestroySemaphore(_device, renderFinishedSem, nullptr);
	vkDestroyFence(_device, inFlightFence, nullptr);

	vkDestroyCommandPool(_device, _commandPool, nullptr);
};

// CPU -> recording command buffer
// GPU -> submitting/executing command buffer
void foto::Render::drawFrame(Device& device, Swapchain& swapchain, Pipeline& pipeline) {
	vkWaitForFences(device.getDevice(), 1, &inFlightFence, VK_TRUE, UINT64_MAX);

	u32 imageIdx;
	vkAcquireNextImageKHR(device.getDevice(), swapchain.getSwapchain(), UINT64_MAX, imageAvailableSem, VK_NULL_HANDLE, &imageIdx);

	vkResetCommandBuffer(commandBuffer, 0);
	recordCommandBuffer(swapchain, pipeline, commandBuffer, imageIdx);

	// Submit info
	VkSemaphore waitSemaphore[] = { imageAvailableSem };
	VkPipelineStageFlags waitStage[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkSemaphore signalSemaphore[] = { renderFinishedSem };

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
	if (vkQueueSubmit(device.getGraphicsQ(), 1, &submitInfo, inFlightFence) != VK_SUCCESS) {
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

	vkQueuePresentKHR(device.getPresentQ(), &presentInfo);
}

void foto::Render::createSyncObjects(VkDevice _device) {
	VkSemaphoreCreateInfo semaphoreCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
	};

	VkFenceCreateInfo fenceCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT
	};

	if (vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &imageAvailableSem) != VK_SUCCESS ||
		vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &renderFinishedSem) != VK_SUCCESS ||
		vkCreateFence(_device, &fenceCreateInfo, nullptr, &inFlightFence) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> createSyncObjects()");
	}
}

void foto::Render::recordCommandBuffer(Swapchain& swapchain,Pipeline& pipeline, VkCommandBuffer commandBuffer, u32 imageIdx) {
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

	VkDependencyInfo toColorDep = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &toColor,
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

	VkRenderingInfo rendering = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.renderArea = {
			.offset = {0, 0},
			.extent = swapchain.getExtent()
		},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachment,
	};
	vkCmdBeginRendering(commandBuffer, &rendering);

	// Bind pipeline/viewport/scissor
	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.getPipeline());

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

	vkCmdDraw(commandBuffer, 3, 1, 0, 0);

	vkCmdEndRendering(commandBuffer);

	// Barrier 2 -> after rendering
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

VkCommandBuffer foto::Render::createCommandBuffer(Device& device, VkCommandPool commandPool) {
	VkCommandBufferAllocateInfo commandBufferAllocInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = commandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};

	VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

	if (vkAllocateCommandBuffers(device.getDevice(), &commandBufferAllocInfo, &commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkRender.cpp -> createCommandBuffer()");
	}

	return commandBuffer;
}