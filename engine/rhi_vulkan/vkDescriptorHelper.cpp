#include "vkDescriptorHelper.hpp"

foto::DescriptorLayoutBuilder& foto::DescriptorLayoutBuilder::addBinding(u32 binding, VkDescriptorType type,
																		 VkShaderStageFlags stages, u32 count) {
	bindings.push_back({
		.binding = binding,
		.descriptorType = type,
		.descriptorCount = count,
		.stageFlags = stages,
	});
	return *this;
}

VkDescriptorSetLayout foto::DescriptorLayoutBuilder::build(Device& device) const {
	VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = static_cast<u32>(bindings.size()),
		.pBindings = bindings.data()
	};

	VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;

	if (vkCreateDescriptorSetLayout(device.getDevice(), &layoutCreateInfo, nullptr, &setLayout) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDescriptorHelper.cpp -> DescriptorLayoutBuilder::build()");
	}

	return setLayout;
}

std::vector<VkDescriptorPoolSize> foto::DescriptorLayoutBuilder::poolSizes(u32 setCount) const {
	// One entry per descriptor type, summed across bindings
	std::vector<VkDescriptorPoolSize> sizes;

	for (const VkDescriptorSetLayoutBinding& binding : bindings) {
		auto existing = std::find_if(sizes.begin(), sizes.end(),
			[&](const VkDescriptorPoolSize& s) { return s.type == binding.descriptorType; });

		if (existing != sizes.end()) {
			existing->descriptorCount += binding.descriptorCount * setCount;
		}
		else {
			sizes.push_back({ binding.descriptorType, binding.descriptorCount * setCount });
		}
	}

	return sizes;
}

foto::DescriptorPool::DescriptorPool(Device& device, u32 maxSets, std::span<const VkDescriptorPoolSize> poolSizes) :
	_device(device.getDevice()), pool(createPool(device, maxSets, poolSizes))
{

};

foto::DescriptorPool::~DescriptorPool() {
	vkDestroyDescriptorPool(_device, pool, nullptr);
};

std::vector<VkDescriptorSet> foto::DescriptorPool::allocate(VkDescriptorSetLayout layout, u32 count) {
	// Same layout for every set, allocate them all in one call
	std::vector<VkDescriptorSetLayout> layouts(count, layout);

	VkDescriptorSetAllocateInfo allocInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = pool,
		.descriptorSetCount = count,
		.pSetLayouts = layouts.data()
	};

	std::vector<VkDescriptorSet> sets(count);
	if (vkAllocateDescriptorSets(_device, &allocInfo, sets.data()) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDescriptorHelper.cpp -> DescriptorPool::allocate()");
	}

	return sets;
}

VkDescriptorPool foto::DescriptorPool::createPool(Device& device, u32 maxSets, std::span<const VkDescriptorPoolSize> poolSizes) {
	VkDescriptorPoolCreateInfo poolCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = maxSets,
		.poolSizeCount = static_cast<u32>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};

	VkDescriptorPool pool = VK_NULL_HANDLE;

	if (vkCreateDescriptorPool(device.getDevice(), &poolCreateInfo, nullptr, &pool) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDescriptorHelper.cpp -> DescriptorPool::createPool()");
	}

	return pool;
}

foto::DescriptorWriter& foto::DescriptorWriter::writeBuffer(u32 binding, VkBuffer buffer, VkDeviceSize size,
															VkDeviceSize offset, VkDescriptorType type) {
	VkDescriptorBufferInfo& info = bufferInfos.emplace_back(VkDescriptorBufferInfo{
		.buffer = buffer,
		.offset = offset,
		.range = size
	});

	writes.push_back({
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = VK_NULL_HANDLE, // filled in by update()
		.dstBinding = binding,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &info
	});
	return *this;
}

foto::DescriptorWriter& foto::DescriptorWriter::writeImage(u32 binding, VkImageView view, VkSampler sampler,
														   VkImageLayout layout, VkDescriptorType type) {
	VkDescriptorImageInfo& info = imageInfos.emplace_back(VkDescriptorImageInfo{
		.sampler = sampler,
		.imageView = view,
		.imageLayout = layout
	});

	writes.push_back({
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = VK_NULL_HANDLE, // filled in by update()
		.dstBinding = binding,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = type,
		.pImageInfo = &info
	});
	return *this;
}

void foto::DescriptorWriter::update(Device& device, VkDescriptorSet set) {
	for (VkWriteDescriptorSet& write : writes) {
		write.dstSet = set;
	}

	vkUpdateDescriptorSets(device.getDevice(), static_cast<u32>(writes.size()), writes.data(), 0, nullptr);
}

void foto::DescriptorWriter::clear() {
	bufferInfos.clear();
	imageInfos.clear();
	writes.clear();
}
