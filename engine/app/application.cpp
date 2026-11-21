#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include <GLFW/glfw3.h>

#include "application.hpp"
#include "../platform/window.hpp"
#include "../rhi_vulkan/vkDevice.hpp"
#include "../rhi_vulkan/vkSwapchain.hpp"
#include "../rhi_vulkan/vkPipeline.hpp"
#include "../rhi_vulkan/vkRender.hpp"
#include "../rhi_vulkan/vkImage.hpp"
#include "../rhi_vulkan/vkDescriptorHelper.hpp"
#include "../assets/gltfLoader.hpp"
#include "../ui/imguiLayer.hpp"

using enginemath::Vec3;
using enginemath::Mat4;

static foto::DescriptorLayoutBuilder sceneSetLayout() {
	return foto::DescriptorLayoutBuilder()
		.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
		.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT,
					foto::MAX_TEXTURES, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
}

foto::Application::Application(const ApplicationConfig& config) :
	window(std::make_unique<Window>(config.width, config.height, config.title)),
	device(std::make_unique<Device>(window->getWindow())),
	swapchain(std::make_unique<Swapchain>(window->getWindow(), *device)),
	pipeline(std::make_unique<Pipeline>(*device, GraphicsPipelineConfig{
				.vertexShaderFilePath = "shaders/basic.vert.spv",
				.fragmentShaderFilePath = "shaders/basic.frag.spv",
				.colorFormat = Render::HDR_FORMAT,
				.depthFormat = Render::DEPTH_FORMAT,
				.descriptorSetLayouts = { sceneSetLayout() },
				.pushConstants = { DRAW_PUSH_RANGE },
			})),
	presentPipeline(std::make_unique<Pipeline>(*device, GraphicsPipelineConfig{
				.vertexShaderFilePath = "shaders/fullscreen.vert.spv",
				.fragmentShaderFilePath = "shaders/tonemap.frag.spv",
				.colorFormat = swapchain->getFormat(),
				.descriptorSetLayouts = { sceneSetLayout() },
				.pushConstants = { DRAW_PUSH_RANGE },
				.useVertexInput = false,
				.cullMode = VK_CULL_MODE_NONE,
				.depthTest = false,
				.depthWrite = false,
			})),
	render(std::make_unique<Render>(*device, *swapchain, *pipeline, *presentPipeline)),
	imgui(std::make_unique<ImGuiLayer>(*window, *device, *swapchain, VK_FORMAT_UNDEFINED)),
	sampler(std::make_unique<Sampler>(*device))
{

}

foto::Application::~Application() {

}

void foto::Application::run() {
	onInit();

	using Clock = std::chrono::steady_clock;
	Clock::time_point lastFrame = Clock::now();

	while (!window->shouldClose()) {
		window->pollEvents();

		Clock::time_point now = Clock::now();
		f32 deltaTime = std::min(std::chrono::duration<f32>(now - lastFrame).count(), 0.1f);
		lastFrame = now;

		drawList.clear();
		imgui->beginFrame();
		onUpdate(deltaTime);
		onImGui();
		imgui->endFrame();

		FrameUbo frameUbo = {
			.view = camera.getView(),
			.proj = camera.getProjection(getAspectRatio()),
			.cameraPos = { camera.position.x, camera.position.y, camera.position.z, 1.0f },
			.lightDir = { lightDirection.x, lightDirection.y, lightDirection.z, 0.0f },
			.exposure = { std::exp2(exposureEV), 0.0f, 0.0f, 0.0f },
			.whiteBalance = { whiteBalance.x, whiteBalance.y, whiteBalance.z, 1.0f }
		};

		bool outOfDate = render->drawFrame(drawList, frameUbo, [this](VkCommandBuffer commandBuffer) {
			imgui->record(commandBuffer);
		});
		if (outOfDate || window->getFramebufferResized()) {
			window->setFramebufferResized(false);
			swapchain->recreate();
			render->onSwapchainRecreated();
		}
	}
	vkDeviceWaitIdle(device->getDevice());
	onShutdown();
}

foto::Mesh* foto::Application::createMesh(std::span<const Vertex> vertices, std::span<const u32> indices, std::vector<Submesh> submeshes) {
	meshes.push_back(std::make_unique<Mesh>(*device, render->getCommandPool(), vertices, indices, std::move(submeshes)));
	return meshes.back().get();
}

foto::Mesh* foto::Application::loadModel(const std::string& filePath) {
	ModelData model = loadGltf(filePath);

	std::vector<u32> textureSlots;
	for (const ImageData& image : model.images) {
		textureSlots.push_back(createTexture(image.rgba.data(), image.width, image.height));
	}

	std::vector<u32> materialSlots;
	for (Material material : model.materials) {
		material.baseColorTexture = (material.baseColorTexture == NO_TEXTURE) ? DEFAULT_TEXTURE : textureSlots.at(material.baseColorTexture);
		materialSlots.push_back(render->addMaterial(material));
	}

	for (Submesh& submesh : model.submeshes) {
		submesh.materialIndex = materialSlots.at(submesh.materialIndex);
	}

	return createMesh(model.vertices, model.indices, std::move(model.submeshes));
}

foto::u32 foto::Application::loadTexture(const std::string& filePath) {
	return addTexture(Image::loadTexture(*device, render->getCommandPool(), filePath));
}

foto::u32 foto::Application::loadTextureHDR(const std::string& filePath) {
	return addTexture(Image::loadTextureHDR(*device, render->getCommandPool(), filePath));
}

foto::u32 foto::Application::createTexture(const void* rgba, u32 width, u32 height) {
	VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * 4;
	return addTexture(Image::createTexture(*device, render->getCommandPool(), rgba, size, { width, height }));
}

foto::u32 foto::Application::addTexture(std::unique_ptr<Image> image) {
	textures.push_back(std::move(image));
	return render->addTexture(*textures.back(), *sampler);
}

foto::u32 foto::Application::createMaterial(const Material& material) {
	return render->addMaterial(material);
}

void foto::Application::updateMaterial(u32 index, const Material& material) {
	render->updateMaterial(index, material);
}

const foto::Material& foto::Application::getMaterial(u32 index) {
	return render->getMaterial(index);
}

void foto::Application::draw(Mesh* mesh, const Mat4& transform, u32 materialOverride, Pipeline* pipeline) {
	drawList.push_back({ .mesh = mesh, .transform = transform, .materialOverride = materialOverride, .pipeline = pipeline });
}

void foto::Application::drawFullscreen(Pipeline* pipeline, u32 material) {
	drawList.push_back({ .mesh = nullptr, .transform = Mat4::identity(), .materialOverride = material, .pipeline = pipeline });
}

foto::Pipeline* foto::Application::createPipeline(GraphicsPipelineConfig config) {
	config.colorFormat = Render::HDR_FORMAT;
	config.depthFormat = Render::DEPTH_FORMAT;
	config.descriptorSetLayouts = { sceneSetLayout() };
	config.pushConstants = { DRAW_PUSH_RANGE };
	pipelines.push_back(std::make_unique<Pipeline>(*device, config));
	return pipelines.back().get();
}

void foto::Application::destroyMesh(Mesh* mesh) {
	auto it = std::find_if(meshes.begin(), meshes.end(), [mesh](const std::unique_ptr<Mesh>& m) { return m.get() == mesh; });
	if (it == meshes.end()) return;
	vkDeviceWaitIdle(device->getDevice());
	meshes.erase(it);
}

void foto::Application::setViewportAspect(f32 aspect) {
	viewportAspect = aspect;
	render->setViewportAspect(aspect);
}

foto::f32 foto::Application::getAspectRatio() {
	if (viewportAspect > 0.0f) return viewportAspect;
	VkExtent2D extent = swapchain->getExtent();
	return static_cast<f32>(extent.width) / static_cast<f32>(std::max(extent.height, 1u));
}

void foto::Application::setLightDirection(const Vec3& direction) {
	lightDirection = direction.normalized();
}

void foto::Application::setExposure(f32 ev) {
	exposureEV = ev;
}

void foto::Application::setWhiteBalance(const Vec3& gains) {
	whiteBalance = gains;
}

void foto::Application::screenshot(const std::string& filePath) {
	render->requestScreenshot(filePath);
}

void foto::Application::flyCamera(f32 deltaTime, f32 speed, f32 sensitivity) {
	bool looking = window->isMouseButtonDown(GLFW_MOUSE_BUTTON_RIGHT) && (window->isCursorCaptured() || !imgui->wantsMouse());
	window->setCursorCaptured(looking);

	if (looking) {
		if (camera.hasOrientation()) camera.clearOrientation();
		enginemath::Vec2 delta = window->getCursorDelta();
		camera.yaw += delta.x * sensitivity;
		camera.pitch -= delta.y * sensitivity;
		camera.clampPitch();
	}

	if (imgui->wantsKeyboard()) return;

	Vec3 move;
	if (window->isKeyDown(GLFW_KEY_W)) move += camera.getForward();
	if (window->isKeyDown(GLFW_KEY_S)) move -= camera.getForward();
	if (window->isKeyDown(GLFW_KEY_D)) move += camera.getRight();
	if (window->isKeyDown(GLFW_KEY_A)) move -= camera.getRight();
	if (window->isKeyDown(GLFW_KEY_E)) move += Vec3::up();
	if (window->isKeyDown(GLFW_KEY_Q)) move -= Vec3::up();

	if (move.dot(move) > 0.0f) {
		f32 boost = window->isKeyDown(GLFW_KEY_LEFT_SHIFT) ? 3.0f : 1.0f;
		camera.position += move.normalized() * (speed * boost * deltaTime);
	}
}
