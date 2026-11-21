#include "vkPipeline.hpp"

foto::Pipeline::Pipeline(Device& device, Swapchain& swapchain) {};

foto::Pipeline::~Pipeline() {};

std::vector<VkShaderModule> foto::Pipeline::createShaderModules(Device& device, const std::vector<std::vector<char>>& code) {
	std::vector<VkShaderModule> shaderModules(code.size());

	for(size i = 0; i < code.size(); i++) {
		VkShaderModuleCreateInfo shaderCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
			.codeSize = code[i].size(),
			.pCode = reinterpret_cast<const u32*>(code[i].data())
		};

		if (vkCreateShaderModule(device.getDevice(), &shaderCreateInfo, nullptr, &shaderModules[i]) != VK_SUCCESS) {
			throw std::runtime_error("ERROR: vkPipeline.cpp -> createShaderModules()");
		}
	}

	return shaderModules;
}
