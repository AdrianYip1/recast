#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"

namespace foto {
	struct GraphicsPipelineConfig {
		std::string vertexShaderFilePath;
		std::string fragmentShaderFilePath;
		VkFormat colorFormat = VK_FORMAT_UNDEFINED;
	};

	class Pipeline {
	public:
		Pipeline(Device& device, const GraphicsPipelineConfig& graphicsPipelineConfig);
		~Pipeline();
		Pipeline(const Pipeline& other) = delete;
		Pipeline& operator=(const Pipeline& other) = delete;

	private:
		static std::vector<char> readFile(const std::string& fileName);
		static std::vector<VkShaderModule> createShaderModules(Device& device, const std::vector<std::vector<char>>& code);

		static VkPipeline createGraphicsPipeline(Device& device, const GraphicsPipelineConfig& config);
	};
} // namespace foto