#pragma once

#include <array>
#include <chrono>
#include <map>
#include <set>

#include <imgui.h>
#include <future>

#include <engine/app/application.hpp>
#include <engine/core/fileWatcher.hpp>

#include "scene/sceneFile.hpp"
#include "lighting/lightFit.hpp"
#include "lighting/cameraMatch.hpp"
#include "scene/raycast.hpp"
#include "pipeline/pythonJob.hpp"

namespace recast {
	// An object photo turned upright and shrunk for drawing its box in the app
	struct PhotoPreview {
		std::filesystem::path file; // a JPEG, empty if the photo couldn't be read
		int width = 0, height = 0;  // the full photo once upright, in pixels
	};
}

class Visualizer : public foto::Application {
public:
	Visualizer(const foto::ApplicationConfig& config, std::string scenePath);

protected:
	void onInit() override;
	void onUpdate(foto::f32 deltaTime) override;
	void onImGui() override;
	void onShutdown() override;

private:
	std::string scenePath;
	recast::SceneFile scene;
	bool sceneLoaded = false;
	bool unsavedChanges = false;
	std::optional<std::chrono::steady_clock::time_point> changedAt; // when the scene first differed from scene.json
	foto::FileWatcher sceneWatcher;

	foto::Pipeline* backgroundPipeline = nullptr;
	foto::Pipeline* compositePipeline = nullptr;
	foto::Pipeline* proxyDepthPipeline = nullptr;
	foto::Pipeline* proxyOverlayPipeline = nullptr;
	foto::u32 photoMaterial = foto::DEFAULT_MATERIAL;
	foto::u32 overlayMaterial = foto::DEFAULT_MATERIAL;
	std::vector<foto::Mesh*> proxyMeshes;
	std::vector<foto::u32> proxyAlbedoMaterials;
	std::string loadedPhotoPath;
	recast::LinearImage photoPixels;
	recast::CameraMatch cameraMatch;

	// Placed objects: scene.objects[i] draws objectMeshOf[i]; copies of one file share its mesh in objectMeshes
	std::map<std::string, foto::Mesh*> objectMeshes;
	std::vector<foto::Mesh*> objectMeshOf;
	std::vector<recast::ObjectTransform> placedTransforms; // where each was when loaded: "reset" goes back there
	int selectedObject = -1;
	bool selectNewestObject = false; // after an object step: the one it added
	bool draggingObject = false;

	// The object bank: every object made so far (data/objects/<entry>/object.glb), ready to add again without Python
	struct BankEntry {
		std::filesystem::path glb;
		std::filesystem::path thumbnail; // the cutout from its photo, empty for objects made before thumbnails
		foto::u64 image = 0;             // thumbnail as an ImGui texture, 0 until loaded
		std::string name;                // its photo's name, under the tile
		std::string label;               // name plus size, in the tooltip
	};
	std::vector<BankEntry> bank;
	// Scenes built so far (data/<photo>_<hash>/scene.json), the drawer's other tab: a click opens one
	struct SceneEntry {
		std::filesystem::path sceneJson;
		std::filesystem::path photo;     // for making its thumbnail
		std::filesystem::path thumbnail; // thumbnail.jpg next to scene.json, made by the engine the first time
		foto::u64 image = 0;
		std::string name, label;
	};
	std::vector<SceneEntry> scenes;
	// Exported images (data/exports/*.png), the drawer's third tab: a click opens one
	struct ExportEntry {
		std::filesystem::path png;
		std::filesystem::path thumbnail; // exports/thumbnails/<name>.jpg, made by the engine the first time
		foto::u64 image = 0;
		std::string name, label;
	};
	std::vector<ExportEntry> exports;
	bool selectExportsTab = false; // after an export: the drawer shows it
	int drawerTab = 0; // 0 objects, 1 scenes, 2 exports
	bool drawerListed = false; // bank and scenes listed at startup, on every scene load, and when their folders change
	std::chrono::steady_clock::time_point drawerChecked;
	std::filesystem::file_time_type bankStamp, dataStamp, exportsStamp;
	std::future<void> thumbnailJob; // scene thumbnails being made in the background
	std::set<std::string> thumbnailTried; // each made once: a photo that can't be read isn't retried every frame
	std::map<std::string, foto::u64> thumbnails; // loaded once per file (ImGui has room for 64)
	bool drawerOpen = true;
	// Bank objects to add on the next update (loading a mesh mid-UI is unsafe): at the cursor ray, or beside the others
	struct PendingAdd {
		std::filesystem::path glb;
		bool atCursor = false;
		enginemath::Vec3 origin, direction;
	};
	std::vector<PendingAdd> pendingAdds;
	bool openCachedScene = false; // set by the scene button when the photo's scene is cached: loaded on the next update
	bool useCache = true; // a photo seen before opens its saved scene, an object photo reuses its bank entry
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

	// The Python pipeline (python/README.md) run from the panel. Its results land in scene.json, which hot reload picks up
	recast::PythonWorker pythonWorker;
	recast::PythonJob pythonJob;
	std::string pythonLabel;
	std::string pipelineStatus;
	bool pythonFailed = false;
	std::string pendingScenePath;
	bool pendingPhotoReload = false;
	bool pendingObjectReload = false;
	std::string followUpLabel;
	std::vector<std::string> followUpArguments;
	bool loadIfFollowUpFails = false;
	std::string pythonCommand(const std::vector<std::string>& arguments) const;
	void startPythonWorker();
	void runPython(const std::string& label, const std::string& module, const std::vector<std::string>& arguments,
				   const std::string& resultScene = "", bool reloadPhoto = false, bool reloadObject = false);
	void finishPython();
	void chooseScenePhoto();
	void chooseObjectPhoto();
	void drawSidebar();
	void drawPrompt();
	float sidebarWidth = 0.0f;
	float drawerHeight = 0.0f; // last frame's, so the prompt card can sit above it
	bool showPlacedTip = false;          // how to move a newly added object

	// The object's box is drawn in the app over a preview of its photo, then sent to Python as --rect (full photo pixels).
	// A photo the app can't read (camera RAW) falls back to Python's own box window
	std::string boxPhoto;                // the photo being boxed, empty when no box is being drawn
	std::future<recast::PhotoPreview> boxJob;
	foto::u64 boxImage = 0;
	ImVec2 boxPhotoSize;                 // the full photo, upright
	ImVec2 boxStart, boxEnd;             // the box in 0-1 of the photo
	bool boxDrawn = false;
	bool boxDragging = false;
	std::array<int, 4> boxPrefill = { 0, 0, 0, 0 }; // the box last used on this photo (x, y, w, h), w = 0 for none
	bool objectBoxedInApp = false;
	std::vector<foto::u64> uiImagesToFree; // freed on the next update, never while the UI that drew them is in flight
	void openBox(const std::string& photo);
	void startObject(const std::string& photo, const std::array<int, 4>* rect);
	void finishBoxPreview();
	void drawBoxSelection();
	void confirmBox();
	void closeBox();

	std::vector<recast::SceneLight> detectedLights; // what "reset lights" goes back to
	std::vector<recast::FitLight> photoEmitters;    // windows and lamps detect found, whose strength the fit solves
	recast::Vec3f detectedAmbient = { 0.0f, 0.0f, 0.0f };
	int selectedLight = -1;
	bool draggingLight = false;
	bool lightMouseWasDown = false;
	enginemath::Vec3 lightPlanePoint;
	enginemath::Vec3 lightPlaneNormal;

	// room editing: boxes and walls from detect (or added) can be dragged, resized, deleted and reset
	bool editRoom = false;
	foto::Mesh* unitBoxMesh = nullptr;
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
	std::map<foto::u32, foto::u32> proxyMaskIds; // proxy id -> its object's cutout number in the owner image
	foto::u32 ownerTexture = 0;
	foto::u32 depthTexture = 0; // the photo's depth (photo_depth.hdr from detect), 0 = none
	std::filesystem::path loadedDepthPath;
	std::filesystem::file_time_type loadedDepthTime;
	std::vector<unsigned char> ownerPixels; // the owner image on the CPU too, for picking (RGBA, R = cutout number)
	int ownerWidth = 0, ownerHeight = 0;
	foto::u32 photoOwnerAt(const enginemath::Vec3& point) const;
	void loadProxyLabels();
	std::string proxyName(size_t index) const;
	bool projectToScreen(const enginemath::Vec3& point, ImVec2& out);
	void drawSegment(ImDrawList* list, enginemath::Vec3 a, enginemath::Vec3 b, ImU32 color, float thickness);
	std::array<enginemath::Vec3, 6> boxHandles(const recast::Proxy& box) const;
	bool wallHandle(const recast::Proxy& wall, enginemath::Vec3& handle) const;
	int boxUnderCursor(const enginemath::Vec3& origin, const enginemath::Vec3& direction) const;
	std::vector<recast::Proxy> roomAtGrab; // every box as it was when a drag started (a whole object moves together)
	bool sameGroup(int a, int b) const;
	int groupSize(int index) const;
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
	foto::Mesh* objectMeshFor(const std::string& path);
	void refreshBank();
	void addFromBank(const PendingAdd& add);
	void loadThumbnails();
	void refreshScenes();
	void refreshExports();
	void drawObjectDrawer();
	bool drawTile(size_t index, foto::u64 image, const std::string& name, const std::string& tooltip, const std::string& badge,
				  const char* dragType);
	void removeObject(int index);
	void drawObjectPanel();
	bool updateLightEditing();
	void addLight();
	void drawLightEditor();
	void drawLightOverlay();
	enginemath::Vec3 markerPosition(const recast::SceneLight& light) const;
	recast::RayHit supportUnderCursor(const enginemath::Vec3& origin, const enginemath::Vec3& direction);
	void drawScene();
	void exportRender();
	enginemath::Mat4 objectTransform(size_t index) const;
};
