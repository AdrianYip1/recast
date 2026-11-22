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
	struct GraphicsPipelineConfig;
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
		u32 loadTextureHDR(const std::string& filePath);
		u32 createTexture(const void* rgba, u32 width, u32 height);
		// Exact values (UNORM, no mips) for lookups like ids: read it with texelFetch
		u32 createDataTexture(const void* rgba, u32 width, u32 height);
		// An image file for the UI: the result is an ImTextureID for ImGui::Image / ImageButton, 0 if it can't be loaded.
		// Each one takes a slot in ImGui's descriptor pool (64 to spare), so callers load each file once
		u64 loadUiImage(const std::string& filePath);
		u32 createMaterial(const Material& material);
		void updateMaterial(u32 index, const Material& material);
		const Material& getMaterial(u32 index);

		// Queues a draw for the current frame; call from onUpdate
		void draw(Mesh* mesh, const enginemath::Mat4& transform, u32 materialOverride = NO_MATERIAL_OVERRIDE, Pipeline* pipeline = nullptr,
				  ShadowCaster shadowCaster = ShadowCaster::None, const enginemath::Vec3& tint = enginemath::Vec3(1.0f, 1.0f, 1.0f));
		void drawFullscreen(Pipeline* pipeline, u32 material);

		// Scene pipeline with the engine's formats, descriptor layout and push constants filled in
		Pipeline* createPipeline(GraphicsPipelineConfig config);
		void destroyMesh(Mesh* mesh);

		// Letterboxes the 3D view to width/height (0 = fill the window)
		void setViewportAspect(f32 aspect);

		Camera& getCamera() { return camera; };
		Window& getWindow() { return *window; };
		f32 getAspectRatio();
		void setLightDirection(const enginemath::Vec3& direction);

		// Up to MAX_LIGHTS point/directional lights with shadows; with none, the light direction above is used instead
		void setLights(std::span<const Light> lights);
		// Region directional light shadow maps cover
		void setShadowBounds(const enginemath::Vec3& center, f32 radius);
		void setAmbient(const enginemath::Vec3& irradiance);
		void screenshot(const std::string& filePath);

		// Renders this frame's draws offscreen at width x height (path traced with `samples` samples when path tracing is on)
		// and saves the tonemapped PNG and the linear .hdr. Call at the end of onUpdate. Blocks until done
		bool exportImage(const std::string& pngPath, const std::string& hdrPath, u32 width, u32 height, u32 samples);

		// Ray traced compositing of the dynamic casters into the photo (photoMaterial's texture), with static casters as
		// the photo's surfaces. Replaces the raster scene: draws without a shadow caster are skipped while it's on
		// The photo's camera, applied to the path traced object: grain in sqrt(linear) units, blur sigma in framebuffer pixels
		void setCameraMatch(f32 grain, f32 blurPixels);
		// Size of the letterboxed 3D view in framebuffer pixels
		enginemath::Vec2 getViewportSize();
		// Top-left of the letterboxed 3D view in framebuffer pixels
		enginemath::Vec2 getViewportOffset();

		bool supportsPathTracing();
		void setPathTracing(bool enabled, u32 photoMaterial);
		// Which cutout each photo pixel shows, so the path tracer can look past boxes where the photo shows something else
		void setPathTraceOwners(u32 ownerTexture);
		// The photo's depth per pixel (distance from the camera in R), so the inserted object hides only behind what the
		// photo really has in front of it. 0 = none (boxes decide)
		void setPathTraceDepth(u32 depthTexture);
		bool isPathTracing();
		// Edge-aware filtering of the path traced noise, strongest right after a change (on by default)
		void setDenoise(bool enabled);
		u32 getPathTraceSamples();

		void setExposure(f32 ev);
		f32 getExposure() { return exposureEV; };
		void setWhiteBalance(const enginemath::Vec3& gains);
		const enginemath::Vec3& getWhiteBalance() { return whiteBalance; };

		// Hold right mouse to look, WASD to move, E/Q up/down, shift to go faster
		void flyCamera(f32 deltaTime, f32 speed = 3.0f, f32 sensitivity = 0.003f);

		// World ray under the mouse; false if the cursor is outside the 3D view or over the UI
		bool getCursorRay(enginemath::Vec3& origin, enginemath::Vec3& direction);
		bool isUiUsingMouse();

	private:
		std::unique_ptr<Window> window;
		std::unique_ptr<Device> device;
		std::unique_ptr<Swapchain> swapchain;
		std::unique_ptr<Pipeline> pipeline;
		std::unique_ptr<Pipeline> presentPipeline;
		std::unique_ptr<Pipeline> shadowPipeline;
		std::unique_ptr<Pipeline> pathTracePipeline;
		std::unique_ptr<Pipeline> causticsPipeline;
		std::unique_ptr<Pipeline> denoisePipeline;
		std::unique_ptr<Render> render;
		std::unique_ptr<ImGuiLayer> imgui;

		std::unique_ptr<Sampler> sampler;
		std::vector<std::unique_ptr<Image>> textures;
		std::vector<std::unique_ptr<Mesh>> meshes;
		std::vector<std::unique_ptr<Pipeline>> pipelines;
		f32 viewportAspect = 0.0f;

		std::vector<DrawItem> drawList;
		Camera camera;
		enginemath::Vec3 lightDirection = enginemath::Vec3(-0.3f, -0.8f, -0.5f).normalized();
		std::vector<Light> lights;
		enginemath::Vec3 shadowCenter;
		f32 shadowRadius = 5.0f;
		enginemath::Vec3 ambient = enginemath::Vec3(0.35f, 0.35f, 0.35f);
		f32 exposureEV = 0.0f;
		f32 cameraGrain = 0.0f;
		f32 cameraBlur = 0.0f;
		enginemath::Vec3 whiteBalance = enginemath::Vec3(1.0f, 1.0f, 1.0f);

		u32 addTexture(std::unique_ptr<Image> image);
		FrameUbo buildFrameUbo();
	};
} // namespace foto
