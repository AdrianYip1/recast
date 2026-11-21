#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "application.hpp"
#include "../platform/window.hpp"
#include "../rhi_vulkan/vkDevice.hpp"
#include "../rhi_vulkan/vkSwapchain.hpp"
#include "../rhi_vulkan/vkPipeline.hpp"
#include "../rhi_vulkan/vkRender.hpp"
#include "../rhi_vulkan/vkBuffer.hpp"
#include "../rhi_vulkan/vkMesh.hpp"
#include "../rhi_vulkan/vkVertex.hpp"
#include "../rhi_vulkan/vkImage.hpp"
#include "../rhi_vulkan/vkDescriptorHelper.hpp"

// pos, normal, uv. +Y is up (proj flips Y), uv (0,0) is the texture's top-left
static const std::vector<foto::Vertex> vertices = {
	{{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
	{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
	{{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}}
};

static const std::vector<foto::u32> indices = {
	0, 1, 2, 2, 3, 0
};

foto::Application::Application(const ApplicationConfig& config) :
	window(std::make_unique<Window>(config.width, config.height, config.title)),
	device(std::make_unique<Device>(window->getWindow())),
	swapchain(std::make_unique<Swapchain>(window->getWindow(), *device)),
	pipeline(std::make_unique<Pipeline>(*device, GraphicsPipelineConfig{
				.vertexShaderFilePath = "shaders/basic.vert.spv",
				.fragmentShaderFilePath = "shaders/basic.frag.spv",
				.colorFormat = swapchain->getFormat(),
				.depthFormat = Render::DEPTH_FORMAT,
				.descriptorSetLayouts = {
					// set 0 matches the bindings in basic.vert / basic.frag
					DescriptorLayoutBuilder()
						.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT)
						.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
				},
			})),
	render(std::make_unique<Render>(*device, *swapchain, *pipeline))
{
	// handle meshes
	quad = std::make_unique<Mesh>(*device, render->getCommandPool(), vertices, indices);

	// handle textures
	texture = Image::loadTexture(*device, render->getCommandPool(), "textures/checker.png");
	sampler = std::make_unique<Sampler>(*device);
	render->setTexture(1, *texture, *sampler);
}

foto::Application::~Application() {

}

void foto::Application::run() {
	onInit();
	while (!window->shouldClose()) {
		window->pollEvents();
		onUpdate(0.0f);

		// Placeholder camera
		Ubo ubo = { enginemath::Mat4::identity(), enginemath::Mat4::identity(), enginemath::Mat4::identity() };
		ubo.proj.m[1][1] = -1.0f;

		Mesh* meshes[] = { quad.get() };
		bool outOfDate = render->drawFrame(meshes, ubo);
		if (outOfDate || window->getFramebufferResized()) {
			window->setFramebufferResized(false);
			swapchain->recreate();
			render->onSwapchainRecreated();
		}
	}
	vkDeviceWaitIdle(device->getDevice());
	onShutdown();
}
