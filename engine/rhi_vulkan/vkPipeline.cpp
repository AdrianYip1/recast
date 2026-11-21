#include "vkPipeline.hpp"

foto::Pipeline::Pipeline(Device& device, const GraphicsPipelineConfig& graphicsPipelineConfig) :
	_device(device.getDevice()),
	shaderCode({ readFile(graphicsPipelineConfig.vertexShaderFilePath), readFile(graphicsPipelineConfig.fragmentShaderFilePath) }),
	setLayoutInfos(graphicsPipelineConfig.descriptorSetLayouts),
	setLayouts(createDescriptorSetLayouts(device, setLayoutInfos)),
	pipelineLayout(createPipelineLayout(device, setLayouts, graphicsPipelineConfig.pushConstants)),
	pipeline(createGraphicsPipeline(device, graphicsPipelineConfig, pipelineLayout, shaderCode))
{
	shaderCode.clear();
};

foto::Pipeline::~Pipeline() {
	vkDestroyPipeline(_device, pipeline, nullptr);
	vkDestroyPipelineLayout(_device, pipelineLayout, nullptr);
	for (VkDescriptorSetLayout setLayout : setLayouts) vkDestroyDescriptorSetLayout(_device, setLayout, nullptr);
};


std::vector<char> foto::Pipeline::readFile(const std::string& fileName) {
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("ERROR: vkPipeline.cpp -> readFile(): can't open " + fileName);
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

std::vector<VkDescriptorSetLayout> foto::Pipeline::createDescriptorSetLayouts(Device& device, const std::vector<DescriptorLayoutBuilder>& infos) {
	std::vector<VkDescriptorSetLayout> setLayouts;
	setLayouts.reserve(infos.size());

	for (const DescriptorLayoutBuilder& info : infos) {
		setLayouts.push_back(info.build(device));
	}

	return setLayouts;
}

VkPipelineLayout foto::Pipeline::createPipelineLayout(Device& device, const std::vector<VkDescriptorSetLayout>& setLayouts,
													  const std::vector<VkPushConstantRange>& pushConstants) {
	// Pipeline Layout
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = static_cast<u32>(setLayouts.size()),
		.pSetLayouts = setLayouts.data(),
		.pushConstantRangeCount = static_cast<u32>(pushConstants.size()),
		.pPushConstantRanges = pushConstants.data()
	};

	if (vkCreatePipelineLayout(device.getDevice(), &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkPipeline.cpp -> createPipelineLayout()");
	}

	return pipelineLayout;
}

VkPipeline foto::Pipeline::createGraphicsPipeline(Device& device, const GraphicsPipelineConfig& config, VkPipelineLayout layout,
												  const std::vector<std::vector<char>>& shaderCode) {

	// Shader Config
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

	VkPipelineShaderStageCreateInfo shaderStagesCreateInfo[] = { vertShaderStageInfo, fragShaderStageInfo };

	// Vertex Input (none when the vertex shader generates its own vertices)

	auto vertexBindingDescription = Vertex::getBindingDescription();
	auto vertexAttributeDescription = Vertex::getAttributeDescriptions();

	VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = config.useVertexInput ? 1u : 0u,
		.pVertexBindingDescriptions = config.useVertexInput ? &vertexBindingDescription : nullptr,
		.vertexAttributeDescriptionCount = config.useVertexInput ? static_cast<u32>(vertexAttributeDescription.size()) : 0u,
		.pVertexAttributeDescriptions = config.useVertexInput ? vertexAttributeDescription.data() : nullptr
	};

	// Input Assembly
	VkPipelineInputAssemblyStateCreateInfo assemblyCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE
	};
	     
	// Viewport + Scissors (Dynamic)
	std::vector<VkDynamicState> dynamicStates = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = static_cast<u32>(dynamicStates.size()),
		.pDynamicStates = dynamicStates.data(),
	};

	VkPipelineViewportStateCreateInfo viewportStateCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1
	};

	// Rasterizer
	if (config.polygonMode != VK_POLYGON_MODE_FILL && !device.supportsWireframe()) {
		throw std::runtime_error("ERROR: vkPipeline.cpp -> createGraphicsPipeline(): polygonMode needs fillModeNonSolid");
	}

	VkPipelineRasterizationStateCreateInfo rasterizerCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.polygonMode = config.polygonMode,
		.cullMode = config.cullMode,
		.frontFace = config.frontFace,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f
	};

	// Multisampling
	VkPipelineMultisampleStateCreateInfo multisamplingCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE
	};

	// Depth (stencil unused)
	bool hasDepth = config.depthFormat != VK_FORMAT_UNDEFINED;

	VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = (hasDepth && config.depthTest) ? VK_TRUE : VK_FALSE,
		.depthWriteEnable = (hasDepth && config.depthWrite) ? VK_TRUE : VK_FALSE,
		.depthCompareOp = config.depthCompare,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
		.minDepthBounds = 0.0f,
		.maxDepthBounds = 1.0f
	};

	// Colour Blending
	VkPipelineColorBlendAttachmentState colorBlendAttachment = {
		.blendEnable = config.alphaBlend ? VK_TRUE : VK_FALSE,
		.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		.colorBlendOp = VK_BLEND_OP_ADD,
		.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
		.alphaBlendOp = VK_BLEND_OP_ADD,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
	};

	VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &colorBlendAttachment,
	};
	
	// Dynamic Rendering
	VkPipelineRenderingCreateInfo renderingCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &config.colorFormat,
		.depthAttachmentFormat = config.depthFormat,
	};
	// Pipeline Creation
	VkPipeline graphicsPipeline = VK_NULL_HANDLE;

	VkGraphicsPipelineCreateInfo pipelineCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &renderingCreateInfo,
		.stageCount = 2,
		.pStages = shaderStagesCreateInfo,
		.pVertexInputState = &vertexInputCreateInfo,
		.pInputAssemblyState = &assemblyCreateInfo,
		.pViewportState = &viewportStateCreateInfo,
		.pRasterizationState = &rasterizerCreateInfo,
		.pMultisampleState = &multisamplingCreateInfo,
		.pDepthStencilState = hasDepth ? &depthStencilCreateInfo : nullptr,
		.pColorBlendState = &colorBlendStateCreateInfo,
		.pDynamicState = &dynamicStateCreateInfo,
		.layout = layout,
		.subpass = 0
		
	};

	if (vkCreateGraphicsPipelines(device.getDevice(), VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkPipeline.cpp -> createGraphicsPipeline()");
	}

	for (VkShaderModule& shaderModule : shaderModules) {
		vkDestroyShaderModule(device.getDevice(), shaderModule, nullptr);
	}

	return graphicsPipeline;
}
