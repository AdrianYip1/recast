#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "application.hpp"
#include "../platform/window.hpp"
#include "../rhi_vulkan/vkDevice.hpp"
#include "../rhi_vulkan/vkSwapchain.hpp"
#include "../rhi_vulkan/vkPipeline.hpp"
#include "../rhi_vulkan/vkRender.hpp"
#include "../rhi_vulkan/vkBuffer.hpp"
#include "../rhi_vulkan/vkVertex.hpp"

static const std::vector<foto::Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

static const std::vector<foto::u16> indices = {
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
			})),
	render(std::make_unique<Render>(*device, *swapchain, *pipeline))
{
	// handle buffers
	VkDeviceSize vertexBytes = sizeof(Vertex) * vertices.size();
	vertexBuffers.push_back(Buffer::createDeviceLocal(*device, render->getCommandPool(),
		vertices.data(), vertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));

	VkDeviceSize indexBytes = sizeof(u16) * indices.size();
	indexBuffer = Buffer::createDeviceLocal(*device, render->getCommandPool(),
		indices.data(), indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
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

		bool outOfDate = render->drawFrame(vertexBuffers, *indexBuffer, ubo);
		if (outOfDate || window->getFramebufferResized()) {
			window->setFramebufferResized(false);
			swapchain->recreate();
			render->onSwapchainRecreated();
		}
	}
	vkDeviceWaitIdle(device->getDevice());
	onShutdown();
}
