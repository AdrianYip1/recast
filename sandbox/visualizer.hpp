#pragma once

#include <chrono>
#include <future>

#include <engine/app/application.hpp>
#include <engine/core/fileWatcher.hpp>

#include "scene/sceneFile.hpp"
#include "lighting/lightFit.hpp"
#include "lighting/cameraMatch.hpp"
#include "scene/raycast.hpp"

class Visualizer : public foto::Application {
public:
	Visualizer(const foto::ApplicationConfig& config, std::string scenePath);

protected:
	void onInit() override;
	void onUpdate(foto::f32 deltaTime) override;
	void onImGui() override;

private:
	std::string scenePath;
	recast::SceneFile scene;
	bool sceneLoaded = false;
	bool unsavedChanges = false;
	foto::FileWatcher sceneWatcher;

	foto::Pipeline* backgroundPipeline = nullptr;
	foto::Pipeline* compositePipeline = nullptr;
	foto::Pipeline* proxyDepthPipeline = nullptr;
	foto::Pipeline* proxyOverlayPipeline = nullptr;
	foto::u32 photoMaterial = foto::DEFAULT_MATERIAL;
	foto::u32 overlayMaterial = foto::DEFAULT_MATERIAL;
	foto::u32 lightMarkerMaterial = foto::DEFAULT_MATERIAL;
	foto::u32 selectedMarkerMaterial = foto::DEFAULT_MATERIAL;
	foto::Mesh* lightMarker = nullptr;
	std::vector<foto::Mesh*> proxyMeshes;
	std::vector<foto::u32> proxyAlbedoMaterials;
	std::string loadedPhotoPath;
	recast::LinearImage photoPixels;
	recast::CameraMatch cameraMatch;

	foto::Mesh* objectMesh = nullptr;
	std::string loadedObjectPath;
	bool draggingObject = false;
	enginemath::Vec3 dragOffset;
	bool leftMouseWasDown = false;
	bool screenshotKeyWasDown = false;

	struct DetectJob {
		std::vector<recast::FitLight> lights;
		recast::LightingFit fit;
		std::string error;
	};
	std::future<DetectJob> detectJob;
	std::chrono::steady_clock::time_point detectStarted;
	std::string lightingStatus;

	enginemath::Vec3 shadowCenter;
	foto::f32 shadowRadius = 5.0f;

	bool exportRequested = false;

	int selectedLight = -1;
	bool draggingLight = false;
	bool lightMouseWasDown = false;
	enginemath::Vec3 lightPlanePoint;
	enginemath::Vec3 lightPlaneNormal;

	bool showProxies = false;
	bool showLights = false;
	bool showOriginal = false;
	bool pathTrace = true;
	bool denoise = true;

	void loadScene();
	void saveScene();
	void applySceneCamera();
	void applySceneLighting();
	void detectLighting();
	void finishLightingDetection();
	void updateObjectInteraction();
	bool updateLightEditing();
	void addLight();
	void drawLightEditor();
	enginemath::Vec3 markerPosition(const recast::SceneLight& light) const;
	recast::RayHit supportUnderCursor(const enginemath::Vec3& origin, const enginemath::Vec3& direction);
	void drawScene();
	void exportRender();
	enginemath::Mat4 objectTransform() const;
};
