#pragma once

#include <array>
#include <chrono>
#include <map>

#include <imgui.h>
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
	// detection that takes longer than this is given up on (its late result is ignored)
	// and the light editor is opened so the lights can be placed by hand
	static constexpr int LIGHT_DETECT_TIMEOUT_S = 10;
	bool detectGaveUp = false;
	bool openLightEditor = false;
	void askForManualLights(const std::string& reason);
	int lightingStride(int base) const;

	enginemath::Vec3 shadowCenter;
	foto::f32 shadowRadius = 5.0f;

	bool exportRequested = false;

	std::vector<recast::SceneLight> detectedLights; // what "reset lights" goes back to
	recast::Vec3f detectedAmbient = { 0.0f, 0.0f, 0.0f };
	int selectedLight = -1;
	bool draggingLight = false;
	bool lightMouseWasDown = false;
	enginemath::Vec3 lightPlanePoint;
	enginemath::Vec3 lightPlaneNormal;

	// room editing: boxes and walls from detect (or added) can be dragged, resized, deleted and reset
	bool editRoom = false;
	foto::Mesh* unitBoxMesh = nullptr;
	foto::u32 selectedProxyMaterial = foto::DEFAULT_MATERIAL;
	std::vector<recast::Vec3f> proxyMeshOrigins;
	std::vector<recast::Proxy> detectedProxies; // what "reset room" goes back to
	int selectedProxy = -1;
	bool draggingProxy = false;
	bool roomMouseWasDown = false;
	bool proxyMeshesDirty = false;
	int hoveredProxy = -1;
	int hoveredHandle = -1;   // 0-5: faces of the selected box (x-, x+, y-, y+, z-, z+), 6: the selected wall
	int draggingHandle = -1;  // -1 while dragging a whole box
	std::map<foto::u32, std::string> proxyLabels;
	void loadProxyLabels();
	std::string proxyName(size_t index) const;
	bool projectToScreen(const enginemath::Vec3& point, ImVec2& out);
	void drawSegment(ImDrawList* list, enginemath::Vec3 a, enginemath::Vec3 b, ImU32 color, float thickness);
	std::array<enginemath::Vec3, 6> boxHandles(const recast::Proxy& box) const;
	bool wallHandle(const recast::Proxy& wall, enginemath::Vec3& handle) const;
	int boxUnderCursor(const enginemath::Vec3& origin, const enginemath::Vec3& direction) const;
	void drawRoomOverlay();
	enginemath::Vec3 proxyGrab;
	recast::Proxy proxyAtGrab;
	void rebuildProxyMeshes();
	enginemath::Mat4 proxyMatrix(size_t index) const;
	bool updateRoomEditing();
	void drawRoomEditor();
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
