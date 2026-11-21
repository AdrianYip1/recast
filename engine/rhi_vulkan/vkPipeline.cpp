#include "vkPipeline.hpp"

foto::Pipeline::Pipeline(Device& device, const GraphicsPipelineConfig& graphicsPipelineConfig) {};

foto::Pipeline::~Pipeline() {};


std::vector<char> foto::Pipeline::readFile(const std::string& fileName) {
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("ERROR: vkPipeline.cpp -> readFile()");
	}

	size fileSize = (size)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

std::vector<VkShaderModule> foto::Pipeline::createShaderModules(Device& device, const std::vector<std::vector<char>>& code) {
	std::vector<VkShaderModule> shaderModules(code.size());

	for(size i = 0; i < code.size(); i++) {
		VkShaderModuleCreateInfo shaderCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = code[i].size(),
			.pCode = reinterpret_cast<const u32*>(code[i].data())
		};

		if (vkCreateShaderModule(device.getDevice(), &shaderCreateInfo, nullptr, &shaderModules[i]) != VK_SUCCESS) {
			throw std::runtime_error("ERROR: vkPipeline.cpp -> createShaderModules()");
		}
	}

	return shaderModules;
}

VkPipeline foto::Pipeline::createGraphicsPipeline(Device& device, const GraphicsPipelineConfig& config) {

	// Shader Config
	std::vector<std::vector<char>> shaderCode = {
		readFile(config.vertexShaderFilePath),
		readFile(config.fragmentShaderFilePath)
	};
	std::vector<VkShaderModule> shaderModules = createShaderModules(device, shaderCode);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = shaderModules[0],
		.pName = "main"
	};
	VkPipelineShaderStageCreateInfo fragShaderStageInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
		.module = shaderModules[1],
		.pName = "main"
	};

	VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

	// Vertex Input

	// Input Assembly

	// Viewport + Scissors (Dynamic)

	// Rasterizer

	// Multisampling

	// Depth + Stencil

	// Colour Blending

	// Pipeline Layout

	//

	for (VkShaderModule& shaderModule : shaderModules) {
		vkDestroyShaderModule(device.getDevice(), shaderModule, nullptr);
	}
}
