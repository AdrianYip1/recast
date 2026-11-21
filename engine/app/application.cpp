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

using enginemath::Vec3;
using enginemath::Mat4;

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
					DescriptorLayoutBuilder()
						.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
						.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
						.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,
									MAX_TEXTURES, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT)
				},
				.pushConstants = { DRAW_PUSH_RANGE },
			})),
	render(std::make_unique<Render>(*device, *swapchain, *pipeline))
{
	// handle meshes
	quad = std::make_unique<Mesh>(*device, render->getCommandPool(), vertices, indices);

	texture = Image::loadTexture(*device, render->getCommandPool(), "textures/checker.png");
	sampler = std::make_unique<Sampler>(*device);
	u32 checkerTexture = render->addTexture(*texture, *sampler);
	checkerMaterial = render->addMaterial({ .baseColorTexture = checkerTexture });
	tintedMaterial = render->addMaterial({ .baseColor = { 1.0f, 0.6f, 0.3f, 1.0f }, .baseColorTexture = checkerTexture });

	camera.position = { 0.6f, 0.4f, 1.6f };
	camera.lookAt({ 0.0f, 0.0f, 0.0f });
}

foto::Application::~Application() {

}

void foto::Application::run() {
	onInit();

	using Clock = std::chrono::steady_clock;
	Clock::time_point lastFrame = Clock::now();

	while (!window->shouldClose()) {
		window->pollEvents();

		// Seconds since the last frame, capped so a hitch (dragging the window, a breakpoint) isn't one giant step
		Clock::time_point now = Clock::now();
		f32 deltaTime = std::min(std::chrono::duration<f32>(now - lastFrame).count(), 0.1f);
		lastFrame = now;

		onUpdate(deltaTime);

		VkExtent2D extent = swapchain->getExtent();
		f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);

		FrameUbo frameUbo = {
			.view = camera.getView(),
			.proj = camera.getProjection(aspect),
			.cameraPos = { camera.position.x, camera.position.y, camera.position.z, 1.0f },
			.lightDir = { -0.3f, -0.8f, -0.5f, 0.0f }
		};
		Vec3 light = Vec3(frameUbo.lightDir.x, frameUbo.lightDir.y, frameUbo.lightDir.z).normalized();
		frameUbo.lightDir = { light.x, light.y, light.z, 0.0f };

		// Two quads, one behind and to the side, to show per-draw transforms + depth testing
		DrawItem items[] = {
			{ .mesh = quad.get(), .transform = Mat4::identity(), .materialOverride = checkerMaterial },
			{ .mesh = quad.get(), .transform = Mat4::translationM(0.45f, 0.2f, -0.4f), .materialOverride = tintedMaterial },
		};
		bool outOfDate = render->drawFrame(items, frameUbo);
		if (outOfDate || window->getFramebufferResized()) {
			window->setFramebufferResized(false);
			swapchain->recreate();
			render->onSwapchainRecreated();
		}
	}
	vkDeviceWaitIdle(device->getDevice());
	onShutdown();
}
