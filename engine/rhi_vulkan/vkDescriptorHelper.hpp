#pragma once

#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include <deque>

#include "vkDevice.hpp"

namespace foto {
	// Describes one descriptor set's bindings (the "shape" shaders expect), builds the VkDescriptorSetLayout.
	//   DescriptorLayoutBuilder()
	//       .addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
	//       .addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
	class DescriptorLayoutBuilder {
	public:
		DescriptorLayoutBuilder& addBinding(u32 binding, VkDescriptorType type, VkShaderStageFlags stages, u32 count = 1);

		// Caller owns the returned layout
		VkDescriptorSetLayout build(Device& device) const;

		// Pool sizes needed to allocate setCount sets of this layout
		std::vector<VkDescriptorPoolSize> poolSizes(u32 setCount) const;

		const std::vector<VkDescriptorSetLayoutBinding>& getBindings() const { return bindings; };

	private:
		std::vector<VkDescriptorSetLayoutBinding> bindings;
	};

	// Owns a VkDescriptorPool. Destroying it frees every set allocated from it.
	class DescriptorPool {
	public:
		DescriptorPool(Device& device, u32 maxSets, std::span<const VkDescriptorPoolSize> poolSizes);
		~DescriptorPool();
		DescriptorPool(const DescriptorPool& other) = delete;
		DescriptorPool& operator=(const DescriptorPool& other) = delete;

		std::vector<VkDescriptorSet> allocate(VkDescriptorSetLayout layout, u32 count);

		VkDescriptorPool getPool() { return pool; };

	private:
		VkDevice _device = VK_NULL_HANDLE;
		VkDescriptorPool pool = VK_NULL_HANDLE;

		static VkDescriptorPool createPool(Device& device, u32 maxSets, std::span<const VkDescriptorPoolSize> poolSizes);
	};

	// Collects descriptor writes, then points a set's bindings at real resources.
	// Reusable: update() can be called on several sets with the same writes.
	// Don't update a set that a command buffer still in flight is using.
	class DescriptorWriter {
	public:
		DescriptorWriter& writeBuffer(u32 binding, VkBuffer buffer, VkDeviceSize size, VkDeviceSize offset = 0,
									  VkDescriptorType type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
		DescriptorWriter& writeImage(u32 binding, VkImageView view, VkSampler sampler,
									 VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
									 VkDescriptorType type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

		void update(Device& device, VkDescriptorSet set);
		void clear();

	private:
		// deque: push_back never moves existing elements, so the writes' pointers stay valid
		std::deque<VkDescriptorBufferInfo> bufferInfos;
		std::deque<VkDescriptorImageInfo> imageInfos;
		std::vector<VkWriteDescriptorSet> writes;
	};
} // namespace foto
