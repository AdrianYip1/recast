#pragma once

#include <engine/app/application.hpp>
#include <engine/core/fileWatcher.hpp>

#include "scene/sceneFile.hpp"

class Visualizer : public foto::Application {
public:
	Visualizer(const foto::ApplicationConfig& config, std::string scenePath);

protected:
	void onInit() override;
	void onUpdate(foto::f32 deltaTime) override;
	void onShutdown() override;
	void onImGui() override;

private:
	foto::Mesh* cube = nullptr;
	foto::Mesh* quad = nullptr;
	foto::Mesh* ground = nullptr;
	foto::u32 checkerMaterial = 0;
	foto::u32 groundMaterial = 0;
	foto::f32 cubeAngle = 0.0f;
	foto::f32 spinSpeed = 0.8f;
	bool screenshotKeyWasDown = false;

	std::string scenePath;
	recast::SceneFile scene;
	bool sceneLoaded = false;
	foto::FileWatcher sceneWatcher;

	foto::Pipeline* backgroundPipeline = nullptr;
	foto::Pipeline* proxyDepthPipeline = nullptr;
	foto::Pipeline* proxyPipeline = nullptr;
	std::string loadedPhotoPath;
	foto::u32 photoMaterial = foto::DEFAULT_MATERIAL;
	foto::u32 proxyMaterial = foto::DEFAULT_MATERIAL;
	std::vector<foto::Mesh*> proxyMeshes;
	bool showProxies = true;
	float proxyOpacity = 0.25f;
	bool freeCamera = false;

	foto::Mesh* objectMesh = nullptr;
	std::string loadedObjectPath;
	bool draggingObject = false;
	enginemath::Vec3 dragOffset;
	bool leftMouseWasDown = false;
	bool unsavedChanges = false;

	void loadScene();
	void applySceneCamera();
	void drawPhotoView();
	void drawDemo();
	void updateObjectInteraction();
	bool cursorHitsObject(const enginemath::Vec3& origin, const enginemath::Vec3& direction);
	enginemath::Mat4 objectTransform() const;
	void saveScene();
	void resolveCollisions();
};
