#pragma once

#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>
#include <engine/scene/camera.hpp>
#include <engine/rhi_vulkan/vkVertex.hpp>
#include <engine/rhi_vulkan/vkMesh.hpp>
#include <engine/rhi_vulkan/vkRenderTypes.hpp>

namespace foto {
	class Window;
	class Device;
	class Swapchain;
	class Pipeline;
	class Render;
	class Image;
	class Sampler;
	class ImGuiLayer;

	struct ApplicationConfig {
		const char* title = "foto";
		u32 width = 1280;
		u32 height = 720;
	};

	class Application {
	public:
		Application(const ApplicationConfig& config);
		virtual ~Application();
		Application(const Application& other) = delete;
		Application& operator=(const Application& other) = delete;

		void run();

	protected:
		virtual void onInit() {}
		virtual void onUpdate(f32 deltaTime) {}
		virtual void onShutdown() {}
		virtual void onImGui() {}

		// Resources live until the application shuts down
		Mesh* createMesh(std::span<const Vertex> vertices, std::span<const u32> indices, std::vector<Submesh> submeshes = {});
		Mesh* loadModel(const std::string& filePath);
		u32 loadTexture(const std::string& filePath);
		u32 createTexture(const void* rgba, u32 width, u32 height);
		u32 createMaterial(const Material& material);
		void updateMaterial(u32 index, const Material& material);
		const Material& getMaterial(u32 index);

		// Queues a draw for the current frame; call from onUpdate
		void draw(Mesh* mesh, const enginemath::Mat4& transform, u32 materialOverride = NO_MATERIAL_OVERRIDE);

		Camera& getCamera() { return camera; };
		Window& getWindow() { return *window; };
		f32 getAspectRatio();
		void setLightDirection(const enginemath::Vec3& direction);

		// Hold right mouse to look, WASD to move, E/Q up/down, shift to go faster
		void flyCamera(f32 deltaTime, f32 speed = 3.0f, f32 sensitivity = 0.003f);

	private:
		std::unique_ptr<Window> window;
		std::unique_ptr<Device> device;
		std::unique_ptr<Swapchain> swapchain;
		std::unique_ptr<Pipeline> pipeline;
		std::unique_ptr<Render> render;
		std::unique_ptr<ImGuiLayer> imgui;

		std::unique_ptr<Sampler> sampler;
		std::vector<std::unique_ptr<Image>> textures;
		std::vector<std::unique_ptr<Mesh>> meshes;

		std::vector<DrawItem> drawList;
		Camera camera;
		enginemath::Vec3 lightDirection = enginemath::Vec3(-0.3f, -0.8f, -0.5f).normalized();

		u32 addTexture(std::unique_ptr<Image> image);
	};
} // namespace foto
