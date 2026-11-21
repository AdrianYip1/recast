#include "vkRender.hpp"
#include "vkImage.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

foto::Render::Render(Device& device, Swapchain& swapchain, Pipeline& pipeline, Pipeline& presentPipeline, Pipeline& shadowPipeline,
					 Pipeline* pathTracePipeline, Pipeline* causticsPipeline, Pipeline* denoisePipeline) :
	device(device), swapchain(swapchain), pipeline(pipeline), presentPipeline(presentPipeline), shadowPipeline(shadowPipeline),
	pathTracePipeline(pathTracePipeline), causticsPipeline(causticsPipeline), denoisePipeline(denoisePipeline),
	_device(device.getDevice()), _commandPool(createCommandPool(device)),
	commandBuffers(createCommandBuffers(device, _commandPool)),
	uniformBuffers(createUniformBuffers(device)),
	// Pool sized from the pipeline's set 0 bindings, one set per frame in flight
	descriptorPool(device, MAX_FRAMES_IN_FLIGHT, pipeline.getDescriptorSetLayoutInfo(0).poolSizes(MAX_FRAMES_IN_FLIGHT),
				   pipeline.getDescriptorSetLayoutInfo(0).updatesAfterBind() ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0u),
	descriptorSets(descriptorPool.allocate(pipeline.getDescriptorSetLayout(0), MAX_FRAMES_IN_FLIGHT)),
	materialBuffers(createMaterialBuffers(device)),
	depthImage(createDepthImage(device, swapchain.getExtent()))
{
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

	if (pathTracePipeline) {
		traceDescriptorPool = std::make_unique<DescriptorPool>(device, MAX_FRAMES_IN_FLIGHT,
															   pathTracePipeline->getDescriptorSetLayoutInfo(1).poolSizes(MAX_FRAMES_IN_FLIGHT));
		traceSets = traceDescriptorPool->allocate(pathTracePipeline->getDescriptorSetLayout(1), MAX_FRAMES_IN_FLIGHT);
		for (size i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			topLevels.push_back(std::make_unique<TopLevelAccelerationStructure>(device, MAX_TRACE_INSTANCES));
			geometryBuffers.push_back(std::make_unique<Buffer>(device, sizeof(GpuGeometry) * MAX_TRACE_GEOMETRIES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
															   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
		}
		createTraceImages();
	}

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
	if (textureCount >= device.getTextureBinding().capacity) {
		throw std::runtime_error("ERROR: vkRender.cpp -> addTexture(): all " + std::to_string(device.getTextureBinding().capacity) + " texture slots used");
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
	if (pathTracePipeline) createTraceImages();
}

void foto::Render::setPathTracing(bool enabled, u32 photoTexture) {
	pathTracing = enabled && pathTracePipeline;
	tracePhotoTexture = photoTexture;
}

void foto::Render::createTraceImages() {
	accumulationImage = std::make_unique<Image>(device, swapchain.getExtent(), VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
	VkExtent2D extent = swapchain.getExtent();
	causticImage = std::make_unique<Image>(device, VkExtent2D{ extent.width * 3, extent.height }, VK_FORMAT_R32_UINT,
										   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	illuminationImage = std::make_unique<Image>(device, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
	guideImage = std::make_unique<Image>(device, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
	for (auto& image : denoiseImages) image = std::make_unique<Image>(device, extent, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT);
	device.immediateSubmit(_commandPool, [&](VkCommandBuffer commandBuffer) {
		for (Image* image : { accumulationImage.get(), causticImage.get(), illuminationImage.get(), guideImage.get(),
							  denoiseImages[0].get(), denoiseImages[1].get() }) {
			Image::transitionLayout(commandBuffer, image->getImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
		}
	});

	for (size i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		DescriptorWriter()
			.writeAccelerationStructure(0, topLevels[i]->getHandle())
			.writeImage(1, accumulationImage->getView(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
			.writeImage(2, hdrImage->getView(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
			.writeBuffer(3, geometryBuffers[i]->getBuffer(), sizeof(GpuGeometry) * MAX_TRACE_GEOMETRIES, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
			.writeImage(4, causticImage->getView(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
			.writeImage(5, illuminationImage->getView(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
			.writeImage(6, guideImage->getView(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
			.writeImage(7, denoiseImages[0]->getView(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
			.writeImage(8, denoiseImages[1]->getView(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
			.update(device, traceSets[i]);
	}
	traceKey.clear();
}

void foto::Render::updateTraceSamples(std::span<const DrawItem> items, const FrameUbo& frameUbo) {
	FrameUbo keyUbo = frameUbo;
	keyUbo.exposure = {};
	keyUbo.whiteBalance = {};

	std::vector<u8> key(sizeof(FrameUbo) + sizeof(u32) * 2);
	std::memcpy(key.data(), &keyUbo, sizeof(FrameUbo));
	std::memcpy(key.data() + sizeof(FrameUbo), &tracePhotoTexture, sizeof(u32));
	u32 version = static_cast<u32>(materialsVersion);
	std::memcpy(key.data() + sizeof(FrameUbo) + sizeof(u32), &version, sizeof(u32));
	for (const DrawItem& item : items) {
		if (!item.mesh || item.shadowCaster == ShadowCaster::None) continue;
		auto append = [&key](const void* data, size bytes) {
			const u8* begin = static_cast<const u8*>(data);
			key.insert(key.end(), begin, begin + bytes);
		};
		append(&item.mesh, sizeof(item.mesh));
		append(&item.transform, sizeof(item.transform));
		append(&item.materialOverride, sizeof(item.materialOverride));
		append(&item.shadowCaster, sizeof(item.shadowCaster));
	}

	if (key != traceKey) {
		traceKey = std::move(key);
		traceSamples = 0;
	}
}

void foto::Render::recordPathTrace(VkCommandBuffer commandBuffer, std::span<const DrawItem> items, const FrameUbo& frameUbo, VkExtent2D extent) {
	updateTraceSamples(items, frameUbo);

	if (traceSamples == 0) {
		VkMemoryBarrier2 beforeClear = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
			.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
		};
		VkDependencyInfo clearDependency = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &beforeClear };
		vkCmdPipelineBarrier2(commandBuffer, &clearDependency);
		VkClearColorValue zero = {};
		VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vkCmdClearColorImage(commandBuffer, causticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
	}

	enginemath::Vec3 objectMin(1e30f), objectMax(-1e30f);
	std::vector<VkAccelerationStructureInstanceKHR> instances;
	std::vector<GpuGeometry> geometries;
	for (const DrawItem& item : items) {
		if (!item.mesh || item.shadowCaster == ShadowCaster::None || !item.mesh->getBottomLevel()) continue;
		std::vector<Submesh>& submeshes = item.mesh->getSubmeshes();
		if (instances.size() >= MAX_TRACE_INSTANCES || geometries.size() + submeshes.size() > MAX_TRACE_GEOMETRIES) break;

		bool dynamic = item.shadowCaster == ShadowCaster::Dynamic;
		if (dynamic) {
			const enginemath::Vec3& low = item.mesh->getBoundsMin();
			const enginemath::Vec3& high = item.mesh->getBoundsMax();
			for (int corner = 0; corner < 8; corner++) {
				enginemath::Vec4 local((corner & 1) ? high.x : low.x, (corner & 2) ? high.y : low.y, (corner & 4) ? high.z : low.z, 1.0f);
				enginemath::Vec4 world = item.transform * local;
				for (int a = 0; a < 3; a++) {
					objectMin[a] = std::min(objectMin[a], world[a]);
					objectMax[a] = std::max(objectMax[a], world[a]);
				}
			}
		}
		VkAccelerationStructureInstanceKHR instance = {
			.instanceCustomIndex = static_cast<u32>(geometries.size()),
			.mask = dynamic ? 0x02u : 0x01u,
			.instanceShaderBindingTableRecordOffset = 0,
			.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
			.accelerationStructureReference = item.mesh->getBottomLevel()->getAddress(),
		};
		for (int row = 0; row < 3; row++) {
			for (int col = 0; col < 4; col++) instance.transform.matrix[row][col] = item.transform.m[col][row];
		}
		instances.push_back(instance);

		for (const Submesh& submesh : submeshes) {
			geometries.push_back({
				.vertexAddress = item.mesh->getVertexBuffer().getDeviceAddress(),
				.indexAddress = item.mesh->getIndexBuffer().getDeviceAddress(),
				.firstIndex = submesh.firstIndex,
				.materialIndex = item.materialOverride != NO_MATERIAL_OVERRIDE ? item.materialOverride : submesh.materialIndex,
				.dynamic = dynamic ? 1u : 0u,
			});
		}
	}
	if (!geometries.empty()) geometryBuffers[currentFrame]->upload(geometries.data(), sizeof(GpuGeometry) * geometries.size());
	topLevels[currentFrame]->build(commandBuffer, instances);

	VkMemoryBarrier2 memoryBarriers[] = {
		{
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
			.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
		},
		{
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT,
			.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		},
	};
	VkImageMemoryBarrier2 hdrToStorage = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		.srcAccessMask = VK_ACCESS_2_NONE,
		.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_GENERAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = hdrImage->getImage(),
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	VkDependencyInfo beforeTrace = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.memoryBarrierCount = 2,
		.pMemoryBarriers = memoryBarriers,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &hdrToStorage,
	};
	vkCmdPipelineBarrier2(commandBuffer, &beforeTrace);

	TracePush push = { .sampleIndex = traceSamples, .photoTexture = tracePhotoTexture, .trace = traceSamples < MAX_TRACE_SAMPLES ? 1u : 0u };
	if (objectMin.x <= objectMax.x) {
		enginemath::Vec3 center = (objectMin + objectMax) * 0.5f;
		enginemath::Vec3 half = (objectMax - objectMin) * 0.5f;
		push.objectSphere = { center.x, center.y, center.z, std::sqrt(half.dot(half)) };
	}
	VkDescriptorSet sets[] = { descriptorSets[currentFrame], traceSets[currentFrame] };

	if (causticsPipeline && push.trace && push.objectSphere.w > 0.0f) {
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, causticsPipeline->getPipeline());
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, causticsPipeline->getPipelineLayout(), 0, 2, sets, 0, nullptr);
		vkCmdPushConstants(commandBuffer, causticsPipeline->getPipelineLayout(), TRACE_PUSH_RANGE.stageFlags, 0, sizeof(TracePush), &push);
		vkCmdDispatch(commandBuffer, (push.photonCount + 63) / 64, 1, 1);

		VkMemoryBarrier2 photonsDone = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
		};
		VkDependencyInfo photonDependency = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &photonsDone };
		vkCmdPipelineBarrier2(commandBuffer, &photonDependency);
	}

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pathTracePipeline->getPipeline());
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pathTracePipeline->getPipelineLayout(), 0, 2, sets, 0, nullptr);
	vkCmdPushConstants(commandBuffer, pathTracePipeline->getPipelineLayout(), TRACE_PUSH_RANGE.stageFlags, 0, sizeof(TracePush), &push);
	vkCmdDispatch(commandBuffer, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);

	if (denoise && denoisePipeline) {
		VkMemoryBarrier2 passDone = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
			.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		};
		VkDependencyInfo passDependency = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &passDone };

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, denoisePipeline->getPipeline());
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, denoisePipeline->getPipelineLayout(), 0, 2, sets, 0, nullptr);
		for (u32 pass = 0; pass < DENOISE_PASSES; pass++) {
			vkCmdPipelineBarrier2(commandBuffer, &passDependency);
			DenoisePush denoisePush = { .pass = pass, .samples = push.trace ? traceSamples + 1 : traceSamples };
			vkCmdPushConstants(commandBuffer, denoisePipeline->getPipelineLayout(), DENOISE_PUSH_RANGE.stageFlags, 0, sizeof(DenoisePush), &denoisePush);
			vkCmdDispatch(commandBuffer, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
		}
	}
	if (traceSamples < MAX_TRACE_SAMPLES) traceSamples++;

	VkImageMemoryBarrier2 hdrToSampled = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
		.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
		.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = hdrImage->getImage(),
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
	};
	VkDependencyInfo afterTrace = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &hdrToSampled };
	vkCmdPipelineBarrier2(commandBuffer, &afterTrace);
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

	if (pathTracing) recordPathTrace(commandBuffer, items, frameUbo, extent);
	else recordRasterScene(commandBuffer, items, frameUbo, extent);

	VkImageMemoryBarrier2 swapchainToAttachment = {
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
	};
	VkDependencyInfo swapchainDependency = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &swapchainToAttachment };
	vkCmdPipelineBarrier2(commandBuffer, &swapchainDependency);

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

void foto::Render::recordRasterScene(VkCommandBuffer commandBuffer, std::span<const DrawItem> items, const FrameUbo& frameUbo, VkExtent2D extent) {
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
	};

	VkDependencyInfo beginDependency = {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.imageMemoryBarrierCount = 2,
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
				.materialIndex = (item.materialOverride != NO_MATERIAL_OVERRIDE) ? item.materialOverride : submesh.materialIndex,
				.tint = { item.tint.x, item.tint.y, item.tint.z, 1.0f }
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
		u32 faceCount = frameUbo.lights[light].intensity.w > 0.5f ? 1 : 6;

		for (u32 face = 0; face < faceCount; face++) {
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
	return std::make_unique<Image>(device, extent, HDR_FORMAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
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

	Mat4 projection = Mat4::projectionM(enginemath::toRad(92.0f), 1.0f, 0.02f, 50.0f);
	for (int face = 0; face < 6; face++) {
		out[face] = projection * Mat4::lookAtM(position, position + directions[face], ups[face]);
	}
}

enginemath::Mat4 foto::shadowDirectionalMatrix(const enginemath::Vec3& direction, const enginemath::Vec3& center, f32 radius) {
	using enginemath::Vec3;
	using enginemath::Mat4;
	Vec3 travel = direction.normalized();
	Vec3 up = std::abs(travel.y) > 0.9f ? Vec3(0, 0, 1) : Vec3(0, 1, 0);
	Mat4 view = Mat4::lookAtM(center - travel * (radius * 2.0f), center, up);

	Mat4 projection = Mat4::zero();
	projection.m[0][0] = 1.0f / radius;
	projection.m[1][1] = 1.0f / radius;
	projection.m[2][2] = -1.0f / (radius * 4.0f);
	projection.m[3][3] = 1.0f;
	return projection * view;
}

std::unique_ptr<foto::Image> foto::Render::createDepthImage(Device& device, VkExtent2D extent) {
	return std::make_unique<Image>(device, extent, DEPTH_FORMAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
								   VK_IMAGE_ASPECT_DEPTH_BIT);
}