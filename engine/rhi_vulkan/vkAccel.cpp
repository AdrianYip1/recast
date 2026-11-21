#include "vkAccel.hpp"
#include "vkMesh.hpp"

static VkDeviceAddress alignAddress(VkDeviceAddress address, VkDeviceSize alignment) {
	return (address + alignment - 1) / alignment * alignment;
}

static std::unique_ptr<foto::Buffer> createScratch(foto::Device& device, VkDeviceSize size) {
	return std::make_unique<foto::Buffer>(device, size + device.getScratchAlignment(),
										  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
										  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

foto::AccelerationStructure::AccelerationStructure(Device& device, VkAccelerationStructureTypeKHR type, VkDeviceSize size) :
	device(device),
	buffer(std::make_unique<Buffer>(device, size,
									VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
									VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
{
	VkAccelerationStructureCreateInfoKHR createInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		.buffer = buffer->getBuffer(),
		.size = size,
		.type = type,
	};
	if (device.getRayTracingFunctions().createAccelerationStructure(device.getDevice(), &createInfo, nullptr, &handle) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkAccel.cpp -> AccelerationStructure()");
	}

	VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
		.accelerationStructure = handle,
	};
	address = device.getRayTracingFunctions().getAddress(device.getDevice(), &addressInfo);
}

foto::AccelerationStructure::~AccelerationStructure() {
	device.getRayTracingFunctions().destroyAccelerationStructure(device.getDevice(), handle, nullptr);
}

std::unique_ptr<foto::AccelerationStructure> foto::AccelerationStructure::buildBottomLevel(Device& device, VkCommandPool commandPool,
																						   Buffer& vertices, u32 vertexCount, Buffer& indices,
																						   std::span<const Submesh> submeshes) {
	const RayTracingFunctions& functions = device.getRayTracingFunctions();

	VkAccelerationStructureGeometryTrianglesDataKHR triangles = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
		.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
		.vertexData = { .deviceAddress = vertices.getDeviceAddress() },
		.vertexStride = sizeof(Vertex),
		.maxVertex = vertexCount - 1,
		.indexType = VK_INDEX_TYPE_UINT32,
		.indexData = { .deviceAddress = indices.getDeviceAddress() },
	};

	std::vector<VkAccelerationStructureGeometryKHR> geometries;
	std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
	std::vector<u32> primitiveCounts;
	for (const Submesh& submesh : submeshes) {
		geometries.push_back({
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
			.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
			.geometry = { .triangles = triangles },
			.flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
		});
		ranges.push_back({ .primitiveCount = submesh.indexCount / 3, .primitiveOffset = submesh.firstIndex * static_cast<u32>(sizeof(u32)) });
		primitiveCounts.push_back(submesh.indexCount / 3);
	}

	VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.geometryCount = static_cast<u32>(geometries.size()),
		.pGeometries = geometries.data(),
	};

	VkAccelerationStructureBuildSizesInfoKHR sizes = { .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	functions.getBuildSizes(device.getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, primitiveCounts.data(), &sizes);

	auto structure = std::make_unique<AccelerationStructure>(device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes.accelerationStructureSize);
	std::unique_ptr<Buffer> scratch = createScratch(device, sizes.buildScratchSize);

	buildInfo.dstAccelerationStructure = structure->getHandle();
	buildInfo.scratchData.deviceAddress = alignAddress(scratch->getDeviceAddress(), device.getScratchAlignment());

	const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = ranges.data();
	device.immediateSubmit(commandPool, [&](VkCommandBuffer commandBuffer) {
		functions.cmdBuild(commandBuffer, 1, &buildInfo, &rangePointer);
	});
	return structure;
}

foto::TopLevelAccelerationStructure::TopLevelAccelerationStructure(Device& device, u32 maxInstances) :
	device(device), maxInstances(maxInstances),
	instanceBuffer(std::make_unique<Buffer>(device, sizeof(VkAccelerationStructureInstanceKHR) * maxInstances,
											VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
											VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
{
	VkAccelerationStructureGeometryKHR geometry = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
		.geometry = { .instances = { .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR } },
	};
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.geometryCount = 1,
		.pGeometries = &geometry,
	};

	VkAccelerationStructureBuildSizesInfoKHR sizes = { .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	device.getRayTracingFunctions().getBuildSizes(device.getDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &maxInstances, &sizes);

	structure = std::make_unique<AccelerationStructure>(device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizes.accelerationStructureSize);
	scratch = createScratch(device, sizes.buildScratchSize);
}

void foto::TopLevelAccelerationStructure::build(VkCommandBuffer commandBuffer, std::span<const VkAccelerationStructureInstanceKHR> instances) {
	u32 count = static_cast<u32>(std::min<size>(instances.size(), maxInstances));
	if (count > 0) instanceBuffer->upload(instances.data(), sizeof(VkAccelerationStructureInstanceKHR) * count);

	VkAccelerationStructureGeometryKHR geometry = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
		.geometry = { .instances = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
			.arrayOfPointers = VK_FALSE,
			.data = { .deviceAddress = instanceBuffer->getDeviceAddress() },
		} },
	};
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
		.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.dstAccelerationStructure = structure->getHandle(),
		.geometryCount = 1,
		.pGeometries = &geometry,
		.scratchData = { .deviceAddress = alignAddress(scratch->getDeviceAddress(), device.getScratchAlignment()) },
	};

	VkAccelerationStructureBuildRangeInfoKHR range = { .primitiveCount = count };
	const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
	device.getRayTracingFunctions().cmdBuild(commandBuffer, 1, &buildInfo, &rangePointer);
}
