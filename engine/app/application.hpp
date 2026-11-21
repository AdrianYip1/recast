#pragma once

#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>
#include <engine/scene/camera.hpp>

namespace foto {
	class Window;
	class Device;
	class Swapchain;
	class Pipeline;
	class Render;
	class Mesh;
	class Image;
	class Sampler;

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
		// Overridden by the game
		virtual void onInit() {}
		virtual void onUpdate(f32 deltaTime) {}
		virtual void onShutdown() {}
		
	private:
		std::unique_ptr<Window> window;
		std::unique_ptr<Device> device;
		std::unique_ptr<Swapchain> swapchain;
		std::unique_ptr<Pipeline> pipeline;
		std::unique_ptr<Render> render;
		std::unique_ptr<Mesh> quad;
		std::unique_ptr<Image> texture;
		std::unique_ptr<Sampler> sampler;
		std::vector<std::unique_ptr<Image>> textures;
		std::vector<std::unique_ptr<Mesh>> meshes;
		Mesh* cube = nullptr;

		Mesh* loadModel(const std::string& filePath);

		Camera camera;
		u32 checkerMaterial = 0;
		u32 tintedMaterial = 0;
	};
} // namespace foto
