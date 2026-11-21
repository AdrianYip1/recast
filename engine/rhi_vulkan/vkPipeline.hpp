#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"
#include "vkDescriptorHelper.hpp"

#include "vkVertex.hpp"

namespace foto {
	struct GraphicsPipelineConfig {
		std::string vertexShaderFilePath;
		std::string fragmentShaderFilePath;
		VkFormat colorFormat = VK_FORMAT_UNDEFINED;

		// One entry per descriptor set: index in the vector = `set = N` in the shaders
		std::vector<DescriptorLayoutBuilder> descriptorSetLayouts;
	};

	class Pipeline {
	public:
		Pipeline(Device& device, const GraphicsPipelineConfig& graphicsPipelineConfig);
		// have overloads for other pipelines
		~Pipeline();
		Pipeline(const Pipeline& other) = delete;
		Pipeline& operator=(const Pipeline& other) = delete;

		VkDescriptorSetLayout getDescriptorSetLayout(u32 set = 0) { return setLayouts.at(set); };
		// Bindings of a set, e.g. to size a descriptor pool for it
		const DescriptorLayoutBuilder& getDescriptorSetLayoutInfo(u32 set = 0) { return setLayoutInfos.at(set); };
		VkPipelineLayout getPipelineLayout() { return pipelineLayout; };
		VkPipeline getPipeline() { return pipeline; };

	private:
		VkDevice _device;
		std::vector<DescriptorLayoutBuilder> setLayoutInfos;
		std::vector<VkDescriptorSetLayout> setLayouts;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipeline pipeline = VK_NULL_HANDLE;

		static std::vector<char> readFile(const std::string& fileName);
		static std::vector<VkShaderModule> createShaderModules(Device& device, const std::vector<std::vector<char>>& code);

		static std::vector<VkDescriptorSetLayout> createDescriptorSetLayouts(Device& device, const std::vector<DescriptorLayoutBuilder>& infos);
		static VkPipelineLayout createPipelineLayout(Device& device, const std::vector<VkDescriptorSetLayout>& setLayouts);
		static VkPipeline createGraphicsPipeline(Device& device, const GraphicsPipelineConfig& config, VkPipelineLayout layout);
	};
} // namespace foto