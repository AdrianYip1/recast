#include "visualizer.hpp"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <engine/platform/window.hpp>
#include <engine/rhi_vulkan/vkPipeline.hpp>

#include "scene/proxyMesh.hpp"
#include "scene/raycast.hpp"
#include "scene/placement.hpp"
#include "ui/theme.hpp"

#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifndef RECAST_PYTHON_DIR
#define RECAST_PYTHON_DIR "python"
#endif
#ifndef RECAST_DATA_DIR
#define RECAST_DATA_DIR "data"
#endif

using enginemath::Vec3;
using enginemath::Mat4;

namespace {
	constexpr const char* PHOTO_FILTER = "Photos\0*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.hdr;*.dng;*.cr2;*.cr3;*.nef;*.arw\0All files\0*.*\0";

	std::string shellQuoted(const std::string& text) {
		return "\"" + text + "\"";
	}

	std::string nativePath(const std::filesystem::path& path) {
		return std::filesystem::path(path).make_preferred().string();
	}

	// The README's venv when it exists, otherwise python from PATH
	std::string pythonExecutable() {
		std::filesystem::path dir = RECAST_PYTHON_DIR;
#ifdef _WIN32
		std::filesystem::path venv = dir / ".venv" / "Scripts" / "python.exe";
#else
		std::filesystem::path venv = dir / ".venv" / "bin" / "python";
#endif
		std::error_code error;
		return std::filesystem::exists(venv, error) ? nativePath(venv) : "python";
	}

	std::string folderName(const std::string& photoPath) {
		std::string name = std::filesystem::path(photoPath).stem().string();
		for (char& c : name) {
			if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') c = '_';
		}
		return name.empty() ? "scene" : name;
	}

	// Short hash of a file's bytes (FNV-1a): the same photo gets the same folder, and so its cached scene
	std::string contentHash(const std::string& path) {
		std::ifstream file(path, std::ios::binary);
		std::uint64_t hash = 1469598103934665603ull;
		std::vector<char> buffer(1 << 16);
		while (true) {
			file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
			std::streamsize count = file.gcount();
			if (count <= 0) break;
			for (std::streamsize i = 0; i < count; i++) {
				hash ^= static_cast<unsigned char>(buffer[static_cast<size_t>(i)]);
				hash *= 1099511628211ull;
			}
		}
		char text[16];
		std::snprintf(text, sizeof(text), "%08llx", static_cast<unsigned long long>((hash >> 32) ^ (hash & 0xffffffffull)));
		return text;
	}

	// A small JPEG of a photo (long side `size`, box-filtered), for the drawer's scene tiles
	bool writeThumbnail(const std::filesystem::path& photo, const std::filesystem::path& out, int size = 160) {
		int width = 0, height = 0, channels = 0;
		unsigned char* pixels = stbi_load(photo.string().c_str(), &width, &height, &channels, 3);
		if (!pixels) return false;
		float step = static_cast<float>(std::max(width, height)) / static_cast<float>(size);
		int w = std::max(1, static_cast<int>(width / step)), h = std::max(1, static_cast<int>(height / step));
		std::vector<unsigned char> small(static_cast<size_t>(w) * h * 3);
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int x0 = static_cast<int>(x * step), x1 = std::min(width, static_cast<int>((x + 1) * step));
				int y0 = static_cast<int>(y * step), y1 = std::min(height, static_cast<int>((y + 1) * step));
				unsigned sum[3] = {}, count = 0;
				for (int sy = y0; sy < y1; sy += 2) {
					for (int sx = x0; sx < x1; sx += 2) {
						const unsigned char* p = pixels + (static_cast<size_t>(sy) * width + sx) * 3;
						for (int c = 0; c < 3; c++) sum[c] += p[c];
						count++;
					}
				}
				for (int c = 0; c < 3; c++) small[(static_cast<size_t>(y) * w + x) * 3 + c] = static_cast<unsigned char>(sum[c] / std::max(count, 1u));
			}
		}
		stbi_image_free(pixels);
		std::filesystem::path temporary = out;
		temporary += ".tmp";
		if (!stbi_write_jpg(temporary.string().c_str(), w, h, 3, small.data(), 85)) return false;
		std::error_code error;
		std::filesystem::rename(temporary, out, error);
		return !error;
	}

	std::string pickPhoto(const char* title) {
		std::string path = recast::openFileDialog(title, PHOTO_FILTER);
		ImGui::GetIO().AddMouseButtonEvent(ImGuiMouseButton_Left, false); // the dialog swallowed the release
		return path;
	}
}

Visualizer::Visualizer(const foto::ApplicationConfig& config, std::string scenePath) :
	Application(config), scenePath(std::move(scenePath))
{
}

void Visualizer::onInit() {
	float dpiX = 1.0f, dpiY = 1.0f;
	if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) glfwGetMonitorContentScale(monitor, &dpiX, &dpiY);
	recast::ui::applyTheme(dpiX, RECAST_ASSETS_DIR);

	backgroundPipeline = createPipeline({
		.vertexShaderFilePath = "shaders/fullscreen.vert.spv",
		.fragmentShaderFilePath = "shaders/background.frag.spv",
		.useVertexInput = false,
		.cullMode = VK_CULL_MODE_NONE,
		.depthTest = false,
		.depthWrite = false,
	});
	compositePipeline = createPipeline({
		.vertexShaderFilePath = "shaders/basic.vert.spv",
		.fragmentShaderFilePath = "shaders/composite.frag.spv",
		.cullMode = VK_CULL_MODE_NONE,
	});
	proxyDepthPipeline = createPipeline({
		.vertexShaderFilePath = "shaders/basic.vert.spv",
		.fragmentShaderFilePath = "shaders/proxy.frag.spv",
		.cullMode = VK_CULL_MODE_NONE,
		.colorWrite = false,
	});
	proxyOverlayPipeline = createPipeline({
		.vertexShaderFilePath = "shaders/basic.vert.spv",
		.fragmentShaderFilePath = "shaders/proxy.frag.spv",
		.cullMode = VK_CULL_MODE_NONE,
		.depthWrite = false,
		.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL,
		.alphaBlend = true,
	});
	overlayMaterial = createMaterial({ .baseColor = { 0.1f, 0.5f, 1.0f, 0.25f } });

	recast::Proxy unitBox;
	unitBox.type = "box";
	unitBox.min = { 0.0f, 0.0f, 0.0f };
	unitBox.max = { 1.0f, 1.0f, 1.0f };
	recast::MeshData unitBoxData = recast::buildProxyMesh(unitBox);
	unitBoxMesh = createMesh(unitBoxData.vertices, unitBoxData.indices);

	startPythonWorker();
	if (!scenePath.empty()) {
		loadScene();
		sceneWatcher = foto::FileWatcher(scenePath);
	}
}

void Visualizer::loadScene() {
	recast::SceneFile loaded;
	try {
		loaded = recast::loadSceneFile(scenePath);
	}
	catch (const std::exception& e) {
		std::cerr << "[recast] " << e.what() << std::endl;
		return;
	}

	std::string photoPath = loaded.resolve(loaded.photo);
	if (photoPath != loadedPhotoPath) {
		try {
			photoMaterial = createMaterial({ .baseColorTexture = loadTextureHDR(photoPath) });
			photoPixels = recast::loadLinearImage(photoPath);
			cameraMatch = recast::estimateCameraMatch(photoPixels);
			loadedPhotoPath = photoPath;
			std::cout << "[recast] photo grain " << cameraMatch.grain << ", blur " << cameraMatch.blur << " px" << std::endl;
		}
		catch (const std::exception& e) {
			std::cerr << "[recast] " << e.what() << std::endl;
		}
	}

	// each object file is loaded once (copies share its mesh); files no object uses any more are dropped
	std::vector<foto::Mesh*> meshes;
	std::set<std::string> used;
	for (const recast::SceneObject& object : loaded.objects) {
		std::string path = loaded.resolve(object.path);
		used.insert(path);
		meshes.push_back(objectMeshFor(path));
	}
	for (auto it = objectMeshes.begin(); it != objectMeshes.end();) {
		if (used.count(it->first)) {
			++it;
			continue;
		}
		if (it->second) destroyMesh(it->second);
		it = objectMeshes.erase(it);
	}
	objectMeshOf = meshes;
	placedTransforms.clear();
	for (const recast::SceneObject& object : loaded.objects) placedTransforms.push_back(object.transform); // "reset" goes back here
	if (selectNewestObject && !loaded.objects.empty()) selectedObject = static_cast<int>(loaded.objects.size()) - 1;
	selectNewestObject = false;
	if (selectedObject >= static_cast<int>(loaded.objects.size())) selectedObject = -1;
	refreshBank();
	refreshScenes();

	scene = std::move(loaded);
	sceneLoaded = true;
	detectedProxies = scene.proxies;
	// the photo's depth from detect: reloaded only when the file changes (each load makes a new texture)
	std::error_code error;
	std::filesystem::path depthPath = scene.directory / "photo_depth.hdr";
	auto depthTime = std::filesystem::last_write_time(depthPath, error);
	if (error) depthTexture = 0;
	else if (depthPath != loadedDepthPath || depthTime != loadedDepthTime) {
		try {
			depthTexture = loadTextureHDR(depthPath.string());
			loadedDepthPath = depthPath;
			loadedDepthTime = depthTime;
		}
		catch (const std::exception& e) {
			std::cerr << "[recast] " << e.what() << std::endl;
			depthTexture = 0;
		}
	}
	selectedProxy = -1;
	hoveredProxy = -1;
	loadProxyLabels();
	draggingProxy = false;
	rebuildProxyMeshes();
	unsavedChanges = false;
	draggingObject = false;
	applySceneCamera();

	if (!photoPixels.rgb.empty()) {
		std::vector<recast::LightingSample> coarse = recast::collectLightingSamples(scene, photoPixels, lightingStride(16));
		if (!coarse.empty()) {
			Vec3 boundsMin(1e30f), boundsMax(-1e30f);
			for (const recast::LightingSample& sample : coarse) {
				for (int a = 0; a < 3; a++) {
					boundsMin[a] = std::min(boundsMin[a], sample.point[a]);
					boundsMax[a] = std::max(boundsMax[a], sample.point[a]);
				}
			}
			shadowCenter = (boundsMin + boundsMax) * 0.5f;
			Vec3 extent = boundsMax - boundsMin;
			shadowRadius = std::max(0.5f * std::sqrt(extent.dot(extent)), 1.0f);
		}
	}

	detectedLights = scene.lights;
	detectedAmbient = scene.lighting.ambient;
	selectedLight = -1;
	photoEmitters = recast::emittersToFit(scene);
	bool unfitted = std::any_of(scene.lights.begin(), scene.lights.end(), [](const recast::SceneLight& light) {
		return light.type == "area" ? light.radiance == recast::Vec3f{ 0, 0, 0 } : light.intensity == recast::Vec3f{ 0, 0, 0 };
	});
	if (scene.lights.empty() || unfitted) detectLighting();
	else lightingStatus = std::to_string(scene.lights.size()) + " light(s) from the saved scene";

	std::cout << "[recast] loaded " << scenePath << std::endl;
}

void Visualizer::saveScene() {
	try {
		recast::saveSceneFile(scene, scenePath);
		sceneWatcher = foto::FileWatcher(scenePath);
		unsavedChanges = false;
	}
	catch (const std::exception& e) {
		std::cerr << "[recast] " << e.what() << std::endl;
	}
}

void Visualizer::applySceneCamera() {
	const recast::SceneCamera& source = scene.camera;
	foto::f32 width = static_cast<foto::f32>(scene.image_size[0]);
	foto::f32 height = static_cast<foto::f32>(scene.image_size[1]);

	Mat4 orientation = Mat4::identity();
	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			orientation.m[col][row] = source.rotation[row][col];
		}
	}

	foto::Camera& camera = getCamera();
	camera.setIntrinsics({ source.fx, source.fy, source.cx, source.cy, width, height });
	camera.setOrientation(orientation);
	camera.position = Vec3(source.position[0], source.position[1], source.position[2]);
	setViewportAspect(width / height);
}

void Visualizer::applySceneLighting() {
	std::vector<foto::Light> lights;
	for (const recast::SceneLight& light : scene.lights) {
		if (light.type == "area") {
			Vec3 c[4];
			for (int k = 0; k < 4; k++) c[k] = Vec3(light.corners[k][0], light.corners[k][1], light.corners[k][2]);
			lights.push_back({
				.type = foto::LightType::Area,
				.position = (c[0] + c[1] + c[2] + c[3]) * 0.25f,
				.direction = (c[1] - c[0]) * 0.5f,
				.intensity = Vec3(light.radiance[0], light.radiance[1], light.radiance[2]),
				.size = 0.5f * (c[3].y - c[0].y),
			});
			continue;
		}
		if (light.type != "point" && light.type != "directional") continue;
		lights.push_back({
			.type = light.type == "directional" ? foto::LightType::Directional : foto::LightType::Point,
			.position = Vec3(light.position[0], light.position[1], light.position[2]),
			.direction = Vec3(light.direction[0], light.direction[1], light.direction[2]),
			.intensity = Vec3(light.intensity[0], light.intensity[1], light.intensity[2]),
			.size = light.type == "directional" ? enginemath::toRad(light.angular_radius_deg) : light.radius,
		});
	}
	setLights(lights);
	setShadowBounds(shadowCenter, shadowRadius);

	const recast::Vec3f& ambient = scene.lighting.ambient;
	setAmbient(Vec3(ambient[0], ambient[1], ambient[2]));
}

void Visualizer::detectLighting() {
	if (detectJob.valid() || loadedPhotoPath.empty()) return;

	lightingStatus = "Detecting lighting...";
	detectStarted = std::chrono::steady_clock::now();
	detectGaveUp = false;
	recast::SceneFile sceneCopy = scene;
	recast::LinearImage photo = photoPixels;
	int stride = lightingStride(8);
	std::vector<recast::FitLight> emitters = photoEmitters;
	detectJob = std::async(std::launch::async, [sceneCopy, photo, stride, emitters]() {
		DetectJob job;
		try {
			std::vector<recast::LightingSample> clipped;
			std::vector<recast::LightingSample> samples = recast::collectLightingSamples(sceneCopy, photo, stride, &clipped);
			job.lights = emitters;
			job.fit = recast::fitMainLight(sceneCopy, samples, job.lights, recast::sunDirections(sceneCopy, photo), clipped);
		}
		catch (const std::exception& e) {
			job.error = e.what();
		}
		return job;
	});
}

// Samples a 2000 px photo at `base`, and bigger photos just as many times, so detection time doesn't grow with resolution
int Visualizer::lightingStride(int base) const {
	int longSide = std::max(photoPixels.width, photoPixels.height);
	return std::max(base, base * longSide / 2000);
}

void Visualizer::askForManualLights(const std::string& reason) {
	lightingStatus = "No light source detected (" + reason + "). Add lights by hand under Show and edit lights.";
	showLights = true;
	openLightEditor = true;
	std::cout << "[recast] no light source detected (" << reason << ")" << std::endl;
}

void Visualizer::finishLightingDetection() {
	if (!detectJob.valid()) return;
	if (detectJob.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
		auto waited = std::chrono::steady_clock::now() - detectStarted;
		if (!detectGaveUp && waited >= std::chrono::seconds(LIGHT_DETECT_TIMEOUT_S)) {
			detectGaveUp = true;
			askForManualLights("gave up after " + std::to_string(LIGHT_DETECT_TIMEOUT_S) + " s");
		}
		return;
	}

	DetectJob job = detectJob.get();
	if (detectGaveUp && !scene.lights.empty()) return; // too late: lights were placed by hand meanwhile
	if (!job.error.empty()) {
		lightingStatus = "Lighting detection failed: " + job.error;
		std::cerr << "[recast] " << lightingStatus << std::endl;
		return;
	}

	scene.lights.clear();
	int windows = 0, others = 0;
	for (size_t i = 0; i < job.lights.size(); i++) {
		scene.lights.push_back(recast::sceneLightFrom(job.lights[i], job.fit.intensities[i]));
		(job.lights[i].area ? windows : others)++;
	}
	scene.lighting.ambient = { job.fit.ambient.x, job.fit.ambient.y, job.fit.ambient.z };
	detectedLights = scene.lights;
	detectedAmbient = scene.lighting.ambient;
	selectedLight = -1;
	unsavedChanges = true;

	int seconds = static_cast<int>(std::chrono::duration<double>(std::chrono::steady_clock::now() - detectStarted).count());
	lightingStatus = "Lit by " + std::to_string(windows) + " window(s) and " + std::to_string(others) + " other light(s), found in " +
					 std::to_string(seconds) + " s";
	std::cout << "[recast] " << lightingStatus << std::endl;
	if (job.lights.empty()) askForManualLights("none found in the photo");
}

void Visualizer::onUpdate(foto::f32 deltaTime) {
	if (sceneWatcher.changed()) loadScene();

	bool screenshotKeyDown = getWindow().isKeyDown(GLFW_KEY_F12);
	if (screenshotKeyDown && !screenshotKeyWasDown) screenshot("screenshot.png");
	screenshotKeyWasDown = screenshotKeyDown;

	finishPython();
	if (openCachedScene) {
		openCachedScene = false;
		loadScene();
		sceneWatcher = foto::FileWatcher(scenePath);
	}
	// listed at startup, then again whenever an object or scene folder appears (the worker writes the example glass,
	// another instance builds a scene); checked twice a second
	auto now = std::chrono::steady_clock::now();
	if (!drawerListed || now - drawerChecked > std::chrono::milliseconds(500)) {
		drawerChecked = now;
		std::error_code error;
		auto bankTime = std::filesystem::last_write_time(std::filesystem::path(RECAST_DATA_DIR) / "objects", error);
		auto dataTime = std::filesystem::last_write_time(std::filesystem::path(RECAST_DATA_DIR), error);
		if (!drawerListed || bankTime != bankStamp) refreshBank();
		if (!drawerListed || dataTime != dataStamp) refreshScenes();
		bankStamp = bankTime;
		dataStamp = dataTime;
		drawerListed = true;
	}
	loadThumbnails();
	if (!sceneLoaded) {
		pendingAdds.clear(); // no room to add objects to yet
		return;
	}
	for (const PendingAdd& add : pendingAdds) addFromBank(add);
	pendingAdds.clear();
	if (proxyMeshesDirty) {
		proxyMeshesDirty = false;
		rebuildProxyMeshes();
	}

	applySceneCamera();
	finishLightingDetection();
	applySceneLighting();
	if (editRoom) updateRoomEditing();
	else if (!(showLights && updateLightEditing())) updateObjectInteraction();
	setPathTraceOwners(ownerTexture);
	setPathTraceDepth(depthTexture);
	// the room and light editors draw over the photo, so the path traced image (the real result) stays on while editing
	setPathTracing(pathTrace && !showOriginal && supportsPathTracing(), photoMaterial);
	float pixelsPerPhotoPixel = getViewportSize().x / std::max(static_cast<float>(scene.image_size[0]), 1.0f);
	setCameraMatch(cameraMatch.grain, cameraMatch.blur * pixelsPerPhotoPixel);
	drawScene();
	if (exportRequested) exportRender();
}

Mat4 Visualizer::objectTransform(size_t index) const {
	return recast::objectMatrix(scene.objects[index].transform);
}

foto::Mesh* Visualizer::objectMeshFor(const std::string& path) {
	auto found = objectMeshes.find(path);
	if (found != objectMeshes.end()) return found->second;
	foto::Mesh* mesh = nullptr;
	try {
		mesh = loadModel(path);
	}
	catch (const std::exception& e) {
		std::cerr << "[recast] " << e.what() << std::endl;
	}
	objectMeshes[path] = mesh;
	return mesh;
}

// Every object made so far, newest first, named after its photo
void Visualizer::refreshBank() {
	bank.clear();
	std::error_code error;
	std::vector<std::pair<std::filesystem::file_time_type, BankEntry>> found;
	for (const auto& folder : std::filesystem::directory_iterator(std::filesystem::path(RECAST_DATA_DIR) / "objects", error)) {
		std::filesystem::path glb = folder.path() / "object.glb", metaPath = folder.path() / "object_meta.json";
		if (!std::filesystem::exists(glb, error) || !std::filesystem::exists(metaPath, error)) continue;
		std::string label = folder.path().filename().string();
		std::string name = label;
		try {
			std::ifstream file(metaPath);
			nlohmann::json meta = nlohmann::json::parse(file);
			std::string photo = std::filesystem::path(meta.value("photo", std::string())).stem().string();
			if (!photo.empty()) {
				name = photo;
				label = photo + (meta.value("glass", false) ? " (glass)" : "");
				if (meta.contains("measurement") && meta["measurement"].contains("height_m")) {
					char size[32];
					std::snprintf(size, sizeof(size), ", %.2f m", meta["measurement"]["height_m"].get<double>());
					label += size;
				}
			}
		}
		catch (const std::exception&) {
		}
		std::filesystem::path thumbnail = folder.path() / "thumbnail.png";
		if (!std::filesystem::exists(thumbnail, error)) thumbnail.clear();
		BankEntry entry{ .glb = glb, .thumbnail = thumbnail, .name = name, .label = label };
		auto loaded = thumbnails.find(thumbnail.string());
		if (!thumbnail.empty() && loaded != thumbnails.end()) entry.image = loaded->second;
		found.push_back({ std::filesystem::last_write_time(glb, error), entry });
	}
	std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	for (auto& [time, entry] : found) bank.push_back(std::move(entry));
}

// Thumbnails for the drawer, loaded in the update (not mid-UI), each file once. A scene without one gets it made from
// its photo, one per frame so opening the drawer never stalls
void Visualizer::loadThumbnails() {
	auto load = [this](const std::filesystem::path& file, foto::u64& image) {
		if (image || file.empty()) return;
		std::string key = file.string();
		auto found = thumbnails.find(key);
		if (found == thumbnails.end()) {
			if (thumbnails.size() >= 60) return;
			found = thumbnails.emplace(key, loadUiImage(key)).first;
		}
		image = found->second;
	};
	for (BankEntry& entry : bank) load(entry.thumbnail, entry.image);

	// scenes without a thumbnail get one made from their photo, in the background (decoding a 12 MP photo takes a while)
	if (thumbnailJob.valid() && thumbnailJob.wait_for(std::chrono::seconds(0)) == std::future_status::ready) thumbnailJob.get();
	std::vector<std::pair<std::filesystem::path, std::filesystem::path>> missing;
	for (SceneEntry& entry : scenes) {
		std::error_code error;
		if (std::filesystem::exists(entry.thumbnail, error)) load(entry.thumbnail, entry.image);
		else if (!entry.photo.empty() && thumbnailTried.insert(entry.thumbnail.string()).second) missing.push_back({ entry.photo, entry.thumbnail });
	}
	if (!missing.empty()) {
		if (thumbnailJob.valid()) thumbnailJob.wait(); // one job at a time (rare: a new scene while the first batch runs)
		thumbnailJob = std::async(std::launch::async, [missing]() {
			for (const auto& [photo, thumbnail] : missing) writeThumbnail(photo, thumbnail);
		});
	}
}

// Every folder under data/ with a scene.json, newest first; the name drops the photo hash the folder carries
void Visualizer::refreshScenes() {
	std::vector<std::pair<std::filesystem::file_time_type, SceneEntry>> found;
	std::error_code error;
	for (const auto& folder : std::filesystem::directory_iterator(std::filesystem::path(RECAST_DATA_DIR), error)) {
		std::filesystem::path json = folder.path() / "scene.json";
		if (!std::filesystem::exists(json, error)) continue;
		SceneEntry entry{ .sceneJson = json, .thumbnail = folder.path() / "thumbnail.jpg" };
		std::string folderName = folder.path().filename().string();
		entry.name = folderName;
		size_t cut = folderName.rfind('_');
		if (cut != std::string::npos && folderName.size() - cut == 9 &&
			std::all_of(folderName.begin() + cut + 1, folderName.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)); })) {
			entry.name = folderName.substr(0, cut);
		}
		entry.label = folderName;
		try {
			std::ifstream file(json);
			nlohmann::json data = nlohmann::json::parse(file);
			entry.photo = folder.path() / data.value("photo", std::string("photo.jpg"));
			size_t objects = data.contains("objects") ? data["objects"].size() : (data.contains("object") && data["object"].is_object() ? 1 : 0);
			if (objects > 0) entry.label += ", " + std::to_string(objects) + " object" + (objects > 1 ? "s" : "");
		}
		catch (const std::exception&) {
			continue;
		}
		auto loaded = thumbnails.find(entry.thumbnail.string());
		if (loaded != thumbnails.end()) entry.image = loaded->second;
		found.push_back({ std::filesystem::last_write_time(json, error), entry });
	}
	std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	scenes.clear();
	for (auto& [time, entry] : found) scenes.push_back(std::move(entry));
}

namespace {
	// A bold heading with a line after it, for each part of the sidebar
	void sectionHeading(const char* text) {
		ImGui::Dummy(ImVec2(0.0f, 0.3f * ImGui::GetFontSize()));
		ImGui::PushFont(recast::ui::boldFont(), 0.0f);
		ImGui::PushStyleColor(ImGuiCol_Text, recast::ui::purple());
		ImGui::SeparatorText(text);
		ImGui::PopStyleColor();
		ImGui::PopFont();
	}

	// A label on the left of the sidebar and the next control filling the rest of the row
	void labelled(const char* label) {
		ImGui::AlignTextToFramePadding();
		ImGui::TextUnformatted(label);
		ImGui::SameLine(0.4f * ImGui::GetWindowWidth());
		ImGui::SetNextItemWidth(-FLT_MIN);
	}

	void hint(const char* text) {
		ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
		ImGui::TextWrapped("%s", text);
		ImGui::PopStyleColor();
	}

	// The width of each of `count` buttons that share the row
	float splitWidth(int count) {
		return (ImGui::GetContentRegionAvail().x - static_cast<float>(count - 1) * ImGui::GetStyle().ItemSpacing.x) / static_cast<float>(count);
	}

	// Drawer tiles grow with the font, so they keep their size on high-DPI screens
	float tileSize() {
		return 4.75f * ImGui::GetFontSize();
	}
}


// One drawer tile: the thumbnail (or the name's first letter), a badge in the corner, the name under it. dragType:
// the tile can be dragged out with its index as that payload. True when clicked
bool Visualizer::drawTile(size_t index, foto::u64 image, const std::string& name, const std::string& tooltip, const std::string& badge,
						  const char* dragType) {
	const float TILE = tileSize();
	const float em = ImGui::GetFontSize();
	const ImGuiStyle& style = ImGui::GetStyle();
	float tileWidth = TILE + 2.0f * style.FramePadding.x;
	ImDrawList* draw = ImGui::GetWindowDrawList();
	ImGui::PushID(static_cast<int>(index));
	ImGui::BeginGroup();
	ImVec2 corner = ImGui::GetCursorScreenPos();
	bool clicked = image
		? ImGui::ImageButton("tile", ImTextureRef(static_cast<ImTextureID>(image)), ImVec2(TILE, TILE),
							 ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f), recast::ui::paperLight())
		: ImGui::Button(name.substr(0, 1).c_str(), ImVec2(tileWidth, TILE + 2.0f * style.FramePadding.y));
	if (dragType && ImGui::BeginDragDropSource()) {
		ImGui::SetDragDropPayload(dragType, &index, sizeof(index));
		if (image) ImGui::Image(ImTextureRef(static_cast<ImTextureID>(image)), ImVec2(3.0f * em, 3.0f * em));
		ImGui::TextUnformatted(name.c_str());
		ImGui::EndDragDropSource();
	}
	else if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip.c_str());
	if (!badge.empty()) {
		ImVec2 size = ImGui::CalcTextSize(badge.c_str());
		ImVec2 at(corner.x + tileWidth - size.x - 0.45f * em, corner.y + 0.3f * em);
		draw->AddRectFilled(ImVec2(at.x - 0.25f * em, at.y - 0.05f * em), ImVec2(at.x + size.x + 0.25f * em, at.y + size.y + 0.05f * em),
							ImGui::GetColorU32(recast::ui::purple()), 0.25f * em);
		draw->AddText(at, IM_COL32_WHITE, badge.c_str());
	}
	// the name under the tile, cut to its width
	ImVec2 textAt = ImGui::GetCursorScreenPos();
	ImGui::Dummy(ImVec2(tileWidth, ImGui::GetTextLineHeight()));
	draw->PushClipRect(textAt, ImVec2(textAt.x + tileWidth, textAt.y + ImGui::GetTextLineHeight()), true);
	draw->AddText(textAt, ImGui::GetColorU32(ImGuiCol_TextDisabled), name.c_str());
	draw->PopClipRect();
	ImGui::EndGroup();
	ImGui::PopID();
	return clicked;
}

// The object bank as a strip of tiles along the bottom, like a game engine's content drawer: click a tile to add the
// object beside the others, or drag it onto the photo to drop it right there. The arrow folds it to a thin bar
void Visualizer::drawObjectDrawer() {
	const float TILE = tileSize();
	const ImGuiStyle& style = ImGui::GetStyle();
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	float bar = ImGui::GetFrameHeight() + 2.0f * style.WindowPadding.y;
	// room for a scrollbar only when the tiles don't fit across
	size_t count = drawerTab == 0 ? bank.size() : scenes.size();
	float strip = static_cast<float>(count) * (TILE + 2.0f * style.FramePadding.x + style.ItemSpacing.x);
	bool overflow = strip > viewport->WorkSize.x - sidebarWidth - 2.0f * style.WindowPadding.x;
	float height = drawerOpen ? bar + TILE + 2.0f * style.FramePadding.y + ImGui::GetTextLineHeightWithSpacing() +
								(overflow ? style.ScrollbarSize : 0.0f) + 0.6f * ImGui::GetFontSize() : bar;
	drawerHeight = height;
	ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + sidebarWidth, viewport->WorkPos.y + viewport->WorkSize.y - height));
	ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x - sidebarWidth, height));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::Begin("##object drawer", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
				 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
	ImGui::PopStyleVar();
	if (ImGui::ArrowButton("##fold", drawerOpen ? ImGuiDir_Down : ImGuiDir_Up)) drawerOpen = !drawerOpen;
	ImGui::SameLine();
	ImGui::PushFont(recast::ui::boldFont(), 0.0f);
	if (ImGui::BeginTabBar("drawer tabs")) {
		std::string objectsTab = "Objects (" + std::to_string(bank.size()) + ")###objects";
		std::string scenesTab = "Scenes (" + std::to_string(scenes.size()) + ")###scenes";
		if (ImGui::BeginTabItem(objectsTab.c_str())) {
			if (drawerTab != 0) drawerOpen = true;
			drawerTab = 0;
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem(scenesTab.c_str())) {
			if (drawerTab != 1) drawerOpen = true;
			drawerTab = 1;
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}
	ImGui::PopFont();

	if (drawerOpen) {
		ImGui::BeginChild("tiles", ImVec2(0.0f, 0.0f), ImGuiChildFlags_None,
						  overflow ? ImGuiWindowFlags_HorizontalScrollbar : ImGuiWindowFlags_NoScrollbar);
		if (drawerTab == 0) {
			std::map<std::string, int> inRoom; // copies of each bank object in the room: a badge on its tile
			for (const recast::SceneObject& object : scene.objects) {
				inRoom[std::filesystem::path(scene.resolve(object.path)).lexically_normal().string()]++;
			}
			if (bank.empty()) hint("Objects you make from a photo show up here.");
			for (size_t i = 0; i < bank.size(); i++) {
				if (i > 0) ImGui::SameLine();
				auto copies = inRoom.find(bank[i].glb.lexically_normal().string());
				std::string badge = copies != inRoom.end() ? "x" + std::to_string(copies->second) : "";
				std::string tooltip = bank[i].label + "\nClick to add, or drag onto the photo";
				if (drawTile(i, bank[i].image, bank[i].name, tooltip, badge, "RECAST_BANK")) pendingAdds.push_back({ .glb = bank[i].glb });
			}
		}
		else {
			std::filesystem::path open = std::filesystem::path(scenePath).lexically_normal();
			for (size_t i = 0; i < scenes.size(); i++) {
				if (i > 0) ImGui::SameLine();
				bool isOpen = scenes[i].sceneJson.lexically_normal() == open;
				if (drawTile(i, scenes[i].image, scenes[i].name, scenes[i].label + (isOpen ? "\nOpen now" : "\nClick to open"),
							 isOpen ? "Open" : "", nullptr) && !isOpen && !pythonJob.running()) {
					if (unsavedChanges) saveScene();
					scenePath = nativePath(scenes[i].sceneJson);
					loadedPhotoPath.clear();
					selectedObject = -1;
					openCachedScene = true;
				}
			}
		}
		ImGui::EndChild();
	}

	// a tile let go over the photo (not over a panel): the object goes where the cursor points
	const ImGuiPayload* payload = ImGui::GetDragDropPayload();
	if (payload && payload->IsDataType("RECAST_BANK") && ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
		!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
		size_t index = *static_cast<const size_t*>(payload->Data);
		PendingAdd add{ .atCursor = true };
		if (index < bank.size() && getCursorRay(add.origin, add.direction)) {
			add.glb = bank[index].glb;
			pendingAdds.push_back(add);
		}
	}
	ImGui::End();
}

// Adds a bank object to the room without Python: on the floor or table in the lower middle of the photo, beside the
// objects already there, facing the camera
void Visualizer::addFromBank(const PendingAdd& add) {
	std::error_code error;
	std::string relative = std::filesystem::relative(add.glb, scene.directory, error).generic_string();
	if (error || relative.empty()) relative = add.glb.generic_string();
	foto::Mesh* mesh = objectMeshFor(scene.resolve(relative));
	if (!mesh) {
		pipelineStatus = "Couldn't load " + add.glb.string();
		return;
	}
	const recast::SceneCamera& camera = scene.camera;
	Vec3 origin(camera.position[0], camera.position[1], camera.position[2]);
	recast::ObjectTransform transform;
	bool placed = false;
	if (add.atCursor) {
		recast::RayHit hit = supportUnderCursor(add.origin, add.direction);
		if (hit.valid()) {
			transform.position = { hit.point.x, hit.point.y, hit.point.z };
			placed = true;
		}
	}
	for (float u : { 0.5f, 0.62f, 0.38f, 0.74f, 0.26f }) {
		if (placed && add.atCursor) break;
		recast::RayHit hit = supportUnderCursor(origin, recast::cameraRay(camera, u * scene.image_size[0], 0.68f * scene.image_size[1]));
		if (!hit.valid()) continue;
		bool free = true;
		for (const recast::SceneObject& other : scene.objects) {
			free = free && std::hypot(other.transform.position[0] - hit.point.x, other.transform.position[2] - hit.point.z) >= 0.25f;
		}
		if (!placed || free) transform.position = { hit.point.x, hit.point.y, hit.point.z };
		placed = true;
		if (free) break;
	}
	if (!placed) transform.position = { shadowCenter.x, 0.0f, shadowCenter.z };
	float toCameraX = origin.x - transform.position[0], toCameraZ = origin.z - transform.position[2];
	transform.rotation_y_deg = std::atan2(toCameraX, toCameraZ) * 180.0f / 3.14159265f;
	recast::resolveCollisions(scene.proxies, transform, mesh->getBoundsMin(), mesh->getBoundsMax());
	scene.objects.push_back({ relative, transform });
	std::cout << "[recast] added " << add.glb.parent_path().filename().string() << (add.atCursor ? " where it was dropped" : " beside the others") << std::endl;
	objectMeshOf.push_back(mesh);
	placedTransforms.push_back(transform);
	selectedObject = static_cast<int>(scene.objects.size()) - 1;
	unsavedChanges = true;
}

void Visualizer::removeObject(int index) {
	if (index < 0 || index >= static_cast<int>(scene.objects.size())) return;
	scene.objects.erase(scene.objects.begin() + index);
	if (index < static_cast<int>(objectMeshOf.size())) objectMeshOf.erase(objectMeshOf.begin() + index);
	if (index < static_cast<int>(placedTransforms.size())) placedTransforms.erase(placedTransforms.begin() + index);
	selectedObject = -1;
	draggingObject = false;
	unsavedChanges = true;
}

// Controls for the object picked in the room (click one); the bank itself is the drawer along the bottom
void Visualizer::drawObjectPanel() {
	bool valid = selectedObject >= 0 && selectedObject < static_cast<int>(scene.objects.size()) &&
				 selectedObject < static_cast<int>(objectMeshOf.size()) && objectMeshOf[selectedObject];
	if (!valid) return;
	const float full = -FLT_MIN;
	std::filesystem::path path(scene.objects[selectedObject].path);
	std::string name = path.has_parent_path() ? path.parent_path().filename().string() : path.stem().string();
	for (const BankEntry& entry : bank) {
		if (entry.glb.lexically_normal() == std::filesystem::path(scene.resolve(scene.objects[selectedObject].path)).lexically_normal()) name = entry.name;
	}
	ImGui::PushFont(recast::ui::boldFont(), 0.0f);
	ImGui::TextColored(recast::ui::purple(), "%s", name.c_str());
	ImGui::PopFont();
	recast::ObjectTransform& transform = scene.objects[selectedObject].transform;
	foto::Mesh* mesh = objectMeshOf[selectedObject];
	labelled("Size");
	if (ImGui::SliderFloat("##size", &transform.scale, 0.1f, 10.0f, "%.2fx", ImGuiSliderFlags_Logarithmic)) {
		transform.scale = std::max(transform.scale, 0.01f);
		recast::resolveCollisions(scene.proxies, transform, mesh->getBoundsMin(), mesh->getBoundsMax());
		unsavedChanges = true;
	}
	const float half = 0.5f * (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x);
	if (ImGui::Button("Reset object", ImVec2(half, 0.0f))) {
		transform = placedTransforms[selectedObject];
		draggingObject = false;
		unsavedChanges = true;
	}
	ImGui::SameLine();
	if (ImGui::Button("Remove", ImVec2(full, 0.0f))) removeObject(selectedObject);
}

// Where the object goes for this mouse ray: the first surface it points at, or the floor in front of it when that's a
// wall or the side of a box, so the object never ends up hidden behind something. A box only counts where the photo's
// cutout says it shows that piece of furniture: elsewhere the box is looser than the real thing, and the ray looks past it
recast::RayHit Visualizer::supportUnderCursor(const Vec3& origin, const Vec3& direction) {
	constexpr float supportNormalY = 0.7f;
	std::vector<recast::Proxy> shown = scene.proxies;
	recast::RayHit hit;
	for (int pass = 0; pass < 8; pass++) {
		hit = recast::raycastProxies(shown, origin, direction);
		if (!hit.valid()) break;
		const recast::Proxy& proxy = shown[hit.proxyIndex];
		auto mask = proxyMaskIds.find(proxy.id);
		if (proxy.type != "box" || mask == proxyMaskIds.end() || photoOwnerAt(hit.point) == mask->second) break;
		shown.erase(shown.begin() + hit.proxyIndex);
	}
	if (!hit.valid() || hit.normal.y >= supportNormalY) return hit;
	return recast::raycastProxies(shown, hit.point + hit.normal * 0.02f, Vec3(0.0f, -1.0f, 0.0f), supportNormalY);
}

// Which detected object's cutout the photo shows where `point` appears (0 = none, or no cutouts)
foto::u32 Visualizer::photoOwnerAt(const Vec3& point) const {
	if (ownerPixels.empty()) return 0;
	const recast::SceneCamera& c = scene.camera;
	Vec3 d = point - Vec3(c.position[0], c.position[1], c.position[2]);
	Vec3 local;
	for (int j = 0; j < 3; j++) local[j] = c.rotation[0][j] * d.x + c.rotation[1][j] * d.y + c.rotation[2][j] * d.z;
	if (local.z > -1e-3f) return 0;
	float u = (c.cx + c.fx * local.x / -local.z) / std::max(1.0f, static_cast<float>(scene.image_size[0]));
	float v = (c.cy - c.fy * local.y / -local.z) / std::max(1.0f, static_cast<float>(scene.image_size[1]));
	if (u < 0.0f || v < 0.0f || u >= 1.0f || v >= 1.0f) return 0;
	int x = std::min(static_cast<int>(u * ownerWidth), ownerWidth - 1), y = std::min(static_cast<int>(v * ownerHeight), ownerHeight - 1);
	return ownerPixels[(static_cast<size_t>(y) * ownerWidth + x) * 4];
}

// Light markers sit at point lights, and 2 m back along a directional light from the middle of the room
Vec3 Visualizer::markerPosition(const recast::SceneLight& light) const {
	if (light.type == "directional") {
		// where the sun is in the sky, seen from the camera (so it sits under the cursor while dragged)
		Vec3 eye(scene.camera.position[0], scene.camera.position[1], scene.camera.position[2]);
		return eye - Vec3(light.direction[0], light.direction[1], light.direction[2]).normalized() * 10.0f;
	}
	return Vec3(light.position[0], light.position[1], light.position[2]);
}

// Click a light marker to select it, drag to move it in the plane facing the camera. True while a light is being dragged
bool Visualizer::updateLightEditing() {
	bool leftDown = getWindow().isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
	bool pressed = leftDown && !lightMouseWasDown && !ImGui::GetIO().WantCaptureMouse;
	lightMouseWasDown = leftDown;
	Vec3 origin, direction;
	bool haveRay = getCursorRay(origin, direction);

	// picked on screen, where the markers are drawn, so a light behind a wall or out past the window is still reachable
	if (pressed && haveRay) {
		ImVec2 mouse = ImGui::GetIO().MousePos;
		float closest = 14.0f * 14.0f;
		int picked = -1;
		for (size_t i = 0; i < scene.lights.size(); i++) {
			if (scene.lights[i].type != "point" && scene.lights[i].type != "directional") continue;
			ImVec2 at;
			if (!projectToScreen(markerPosition(scene.lights[i]), at)) continue;
			float dx = at.x - mouse.x, dy = at.y - mouse.y;
			if (dx * dx + dy * dy < closest) {
				closest = dx * dx + dy * dy;
				picked = static_cast<int>(i);
			}
		}
		if (picked >= 0) {
			selectedLight = picked;
			draggingLight = true;
			lightPlanePoint = markerPosition(scene.lights[picked]);
			lightPlaneNormal = direction;
		}
	}
	if (!leftDown) draggingLight = false;

	// scrolling moves the selected point light nearer or further along its line from the camera
	float scroll = getWindow().getScrollDelta();
	if (scroll != 0.0f && selectedLight >= 0 && selectedLight < static_cast<int>(scene.lights.size()) &&
		scene.lights[selectedLight].type == "point" && !ImGui::GetIO().WantCaptureMouse) {
		recast::SceneLight& light = scene.lights[selectedLight];
		const recast::Vec3f& eye = scene.camera.position;
		for (int a = 0; a < 3; a++) light.position[a] = eye[a] + (light.position[a] - eye[a]) * std::pow(1.15f, -scroll);
		unsavedChanges = true;
	}

	if (draggingLight && haveRay && selectedLight >= 0 && selectedLight < static_cast<int>(scene.lights.size())) {
		float facing = direction.dot(lightPlaneNormal);
		float t = std::abs(facing) > 1e-4f ? (lightPlanePoint - origin).dot(lightPlaneNormal) / facing : -1.0f;
		if (t > 0.0f) {
			Vec3 at = origin + direction * t;
			recast::SceneLight& light = scene.lights[selectedLight];
			if (light.type == "directional") {
				// the sun goes where the cursor points, kept above the horizon
				Vec3 sky = direction;
				sky.y = std::max(sky.y, 0.05f);
				sky.normalize();
				light.direction = { -sky.x, -sky.y, -sky.z };
			}
			else {
				light.position = { at.x, at.y, at.z };
			}
			unsavedChanges = true;
		}
	}
	return draggingLight;
}

// Markers over the photo, never hidden by walls: a dot per light (named, selected one larger), windows as outlines
void Visualizer::drawLightOverlay() {
	ImDrawList* list = ImGui::GetForegroundDrawList();
	int number = 0;
	for (size_t i = 0; i < scene.lights.size(); i++) {
		const recast::SceneLight& light = scene.lights[i];
		bool selected = static_cast<int>(i) == selectedLight;
		ImU32 color = selected ? IM_COL32(110, 220, 255, 255) : IM_COL32(255, 220, 80, 255);
		if (light.type == "area") {
			for (int k = 0; k < 4; k++) {
				const recast::Vec3f& a = light.corners[k];
				const recast::Vec3f& b = light.corners[(k + 1) % 4];
				drawSegment(list, Vec3(a[0], a[1], a[2]), Vec3(b[0], b[1], b[2]), IM_COL32(255, 220, 80, 160), 1.5f);
			}
			ImVec2 at;
			Vec3 centre(0.0f);
			for (const recast::Vec3f& c : light.corners) centre = centre + Vec3(c[0], c[1], c[2]) * 0.25f;
			if (projectToScreen(centre, at)) list->AddText(ImVec2(at.x - 36.0f, at.y), IM_COL32(255, 220, 80, 220), "window light");
			continue;
		}
		if (light.type != "point" && light.type != "directional") continue;
		number++;
		ImVec2 at;
		if (!projectToScreen(markerPosition(light), at)) continue;
		float radius = selected ? 10.0f : 8.0f;
		list->AddCircleFilled(at, radius, color);
		list->AddCircle(at, radius, IM_COL32(0, 0, 0, 220), 0, 2.0f);
		std::string name = light.type == "directional" ? "sun" : "light " + std::to_string(number);
		list->AddText(ImVec2(at.x + radius + 4.0f, at.y - 7.0f), color, name.c_str());
	}
}

// A point light 1.2 m above the object, as strong as the first point light (or a neutral one)
void Visualizer::addLight() {
	recast::SceneLight light;
	light.type = "point";
	int nearObject = selectedObject >= 0 && selectedObject < static_cast<int>(scene.objects.size()) ? selectedObject : (scene.objects.empty() ? -1 : 0);
	Vec3 above = shadowCenter + Vec3(0.0f, 1.2f, 0.0f);
	if (nearObject >= 0) {
		const recast::Vec3f& base = scene.objects[nearObject].transform.position;
		above = Vec3(base[0], base[1] + 1.2f, base[2]);
	}
	light.position = { above.x, above.y, above.z };
	light.intensity = { 2.0f, 2.0f, 2.0f };
	for (const recast::SceneLight& existing : scene.lights) {
		if (existing.type == "point") {
			light.intensity = existing.intensity;
			break;
		}
	}
	scene.lights.push_back(light);
	selectedLight = static_cast<int>(scene.lights.size()) - 1;
	unsavedChanges = true;
}

void Visualizer::drawLightEditor() {
	hint("Drag a light to move it. Scroll over a light to move it nearer or further.");
	bool canAdd = scene.lights.size() < foto::MAX_LIGHTS;
	int buttons = (canAdd ? 2 : 0) + (detectedLights.empty() ? 0 : 1);
	if (buttons > 0) {
		const float width = splitWidth(buttons);
		if (canAdd) {
			if (ImGui::Button("Add light", ImVec2(width, 0.0f))) addLight();
			ImGui::SameLine();
			if (ImGui::Button("Add sun", ImVec2(width, 0.0f))) {
				recast::SceneLight sun;
				sun.type = "directional";
				Vec3 travels = Vec3(0.3f, -0.5f, 0.8f).normalized(); // from the far side of the room, 30 degrees up
				sun.direction = { travels.x, travels.y, travels.z };
				sun.intensity = { 2.0f, 1.9f, 1.7f };
				for (const recast::SceneLight& existing : scene.lights) {
					if (existing.type == "directional") sun.intensity = existing.intensity;
				}
				scene.lights.push_back(sun);
				selectedLight = static_cast<int>(scene.lights.size()) - 1;
				unsavedChanges = true;
			}
		}
		if (!detectedLights.empty()) {
			if (canAdd) ImGui::SameLine();
			if (ImGui::Button("Reset lights", ImVec2(width, 0.0f))) {
				scene.lights = detectedLights;
				scene.lighting.ambient = detectedAmbient;
				selectedLight = -1;
				unsavedChanges = true;
			}
		}
	}

	if (selectedLight < 0 || selectedLight >= static_cast<int>(scene.lights.size())) return;
	recast::SceneLight& light = scene.lights[selectedLight];
	ImGui::PushFont(recast::ui::boldFont(), 0.0f);
	ImGui::TextColored(recast::ui::purple(), "%s", light.type == "directional" ? "Sun" : ("Light " + std::to_string(selectedLight + 1)).c_str());
	ImGui::PopFont();

	if (light.type == "directional") {
		// where the sun is in the sky: around (0 = straight ahead of the camera, positive = to its right) and above the
		// horizon, in degrees
		const auto& r = scene.camera.rotation;
		Vec3 ahead = Vec3(-r[0][2], 0.0f, -r[2][2]);
		ahead = ahead.dot(ahead) > 1e-6f ? ahead.normalized() : Vec3(0.0f, 0.0f, -1.0f);
		Vec3 right = ahead.cross(Vec3(0.0f, 1.0f, 0.0f));
		Vec3 sky = -Vec3(light.direction[0], light.direction[1], light.direction[2]).normalized();
		float height = enginemath::toDeg(std::asin(std::clamp(sky.y, -1.0f, 1.0f)));
		float around = enginemath::toDeg(std::atan2(sky.dot(right), sky.dot(ahead)));
		labelled("Direction");
		bool changed = ImGui::SliderFloat("##sun direction", &around, -180.0f, 180.0f, "%.0f deg");
		labelled("Height");
		changed |= ImGui::SliderFloat("##sun height", &height, 2.0f, 89.0f, "%.0f deg");
		if (changed) {
			float h = enginemath::toRad(height), a = enginemath::toRad(around);
			Vec3 toSun = (ahead * std::cos(a) + right * std::sin(a)) * std::cos(h) + Vec3(0.0f, std::sin(h), 0.0f);
			light.direction = { -toSun.x, -toSun.y, -toSun.z };
			unsavedChanges = true;
		}
	}

	float strongest = std::max({ light.intensity[0], light.intensity[1], light.intensity[2] });
	float brightness = strongest;
	labelled("Brightness");
	if (ImGui::SliderFloat("##brightness", &brightness, 0.0f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic)) {
		for (float& channel : light.intensity) channel = strongest > 1e-6f ? channel * brightness / strongest : brightness;
		unsavedChanges = true;
	}

	// a sun shines from where the light is, towards the middle of the room; a point light goes back to that spot
	const float half = splitWidth(2);
	if (ImGui::Button(light.type == "point" ? "Make it a sun" : "Make it a point light", ImVec2(half, 0.0f))) {
		if (light.type == "point") {
			Vec3 from(light.position[0], light.position[1], light.position[2]);
			Vec3 toCenter = shadowCenter - from;
			if (toCenter.dot(toCenter) > 1e-6f) {
				toCenter.normalize();
				light.type = "directional";
				light.direction = { toCenter.x, toCenter.y, toCenter.z };
				// a point light's intensity falls off with distance; the sun keeps what it gave the middle of the room
				float d2 = (shadowCenter - from).dot(shadowCenter - from);
				for (float& channel : light.intensity) channel /= std::max(d2, 1e-3f);
			}
		}
		else {
			Vec3 at = markerPosition(light);
			float d2 = (shadowCenter - at).dot(shadowCenter - at);
			light.type = "point";
			light.position = { at.x, at.y, at.z };
			for (float& channel : light.intensity) channel *= d2;
		}
		unsavedChanges = true;
	}
	ImGui::SameLine();
	if (ImGui::Button("Remove light", ImVec2(half, 0.0f))) {
		scene.lights.erase(scene.lights.begin() + selectedLight);
		selectedLight = -1;
		unsavedChanges = true;
	}
}

void Visualizer::updateObjectInteraction() {
	foto::Window& window = getWindow();
	bool leftDown = window.isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
	Vec3 origin, direction;
	bool haveRay = getCursorRay(origin, direction);

	// a click selects the nearest object under the cursor and starts dragging it
	if (leftDown && !leftMouseWasDown && haveRay) {
		float nearest = 1e30f;
		int picked = -1;
		for (size_t i = 0; i < scene.objects.size() && i < objectMeshOf.size(); i++) {
			if (!objectMeshOf[i]) continue;
			Vec3 worldMin, worldMax;
			recast::objectWorldBounds(scene.objects[i].transform, objectMeshOf[i]->getBoundsMin(), objectMeshOf[i]->getBoundsMax(), worldMin, worldMax);
			float t = 0.0f;
			if (recast::rayAabb(origin, direction, worldMin, worldMax, t) && t < nearest) {
				nearest = t;
				picked = static_cast<int>(i);
			}
		}
		recast::RayHit hit = supportUnderCursor(origin, direction);
		if (picked >= 0) {
			selectedObject = picked;
			if (hit.valid()) {
				const recast::Vec3f& position = scene.objects[picked].transform.position;
				draggingObject = true;
				dragOffset = Vec3(position[0] - hit.point.x, 0.0f, position[2] - hit.point.z);
			}
		}
	}
	if (!leftDown) draggingObject = false;

	if (selectedObject < 0 || selectedObject >= static_cast<int>(scene.objects.size()) ||
		selectedObject >= static_cast<int>(objectMeshOf.size()) || !objectMeshOf[selectedObject]) {
		draggingObject = false;
		leftMouseWasDown = leftDown;
		return;
	}
	recast::ObjectTransform& transform = scene.objects[selectedObject].transform;
	foto::Mesh* objectMesh = objectMeshOf[selectedObject];

	bool moved = false;
	if (draggingObject && haveRay) {
		recast::RayHit hit = supportUnderCursor(origin, direction);
		if (hit.valid()) {
			transform.position = { hit.point.x + dragOffset.x, hit.point.y, hit.point.z + dragOffset.z };
			moved = true;
		}
	}

	float scroll = window.getScrollDelta();
	if (scroll != 0.0f && haveRay) {
		if (window.isKeyDown(GLFW_KEY_LEFT_SHIFT)) transform.scale *= std::pow(1.1f, scroll);
		else transform.rotation_y_deg = std::fmod(transform.rotation_y_deg + scroll * 15.0f + 360.0f, 360.0f);
		moved = true;
	}

	if (moved) {
		recast::resolveCollisions(scene.proxies, transform, objectMesh->getBoundsMin(), objectMesh->getBoundsMax());
		unsavedChanges = true;
	}
	leftMouseWasDown = leftDown;
}

// Renders the composite at the photo's resolution into renders/ next to scene.json and opens it
void Visualizer::exportRender() {
	exportRequested = false;
	std::time_t now = std::time(nullptr);
	char stamp[32];
	std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));

	std::filesystem::path folder = scene.directory / "renders";
	std::filesystem::create_directories(folder);
	std::string name = std::filesystem::path(scenePath).stem().string() + "_" + stamp;
	std::string pngPath = (folder / (name + ".png")).string();
	std::string hdrPath = (folder / (name + ".hdr")).string();

	setCameraMatch(cameraMatch.grain, cameraMatch.blur);
	auto started = std::chrono::steady_clock::now();
	bool written = exportImage(pngPath, hdrPath, scene.image_size[0], scene.image_size[1], isPathTracing() ? 512 : 1);
	int seconds = static_cast<int>(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());

	if (!written) {
		lightingStatus = "Export failed: couldn't write " + pngPath;
		return;
	}
	lightingStatus = "Exported " + pngPath + " in " + std::to_string(seconds) + " s";
	std::cout << "[recast] " << lightingStatus << std::endl;
#ifdef _WIN32
	std::system(("start \"\" \"" + pngPath + "\"").c_str());
#else
	std::system(("xdg-open \"" + pngPath + "\" &").c_str());
#endif
}

void Visualizer::drawScene() {
	if (!loadedPhotoPath.empty()) drawFullscreen(backgroundPipeline, photoMaterial);

	for (size_t i = 0; i < proxyMeshes.size(); i++) {
		if (!proxyMeshes[i]) continue;
		Mat4 matrix = proxyMatrix(i);
		if (showOriginal) draw(proxyMeshes[i], matrix, overlayMaterial, proxyDepthPipeline, foto::ShadowCaster::Static);
		else if (isPathTracing()) draw(proxyMeshes[i], matrix, proxyAlbedoMaterials[i], nullptr, foto::ShadowCaster::Static);
		// the raster preview's shadow maps have no window openings: walls cast none there, so sunlight through them shows
		else draw(proxyMeshes[i], matrix, photoMaterial, compositePipeline,
				  scene.proxies[i].type == "plane" && std::abs(scene.proxies[i].normal[1]) < 0.5f ? foto::ShadowCaster::None : foto::ShadowCaster::Static);
	}

	if (showOriginal) return;
	for (size_t i = 0; i < scene.objects.size() && i < objectMeshOf.size(); i++) {
		if (objectMeshOf[i]) draw(objectMeshOf[i], objectTransform(i), foto::NO_MATERIAL_OVERRIDE, nullptr, foto::ShadowCaster::Dynamic);
	}
}

void Visualizer::onImGui() {
	drawSidebar();
	drawPrompt();
	// the start is only the welcome card: the drawer comes in with the first scene
	if (sceneLoaded) drawObjectDrawer();
	else drawerHeight = 0.0f;
}

// Everything the app does, in a sidebar that always fills the left of the window
void Visualizer::drawSidebar() {
	// hidden until there's a scene and an object to work on: the prompts and the drawer lead up to that
	if (!sceneLoaded || scene.objects.empty()) {
		sidebarWidth = 0.0f;
		return;
	}
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	sidebarWidth = std::min(22.0f * ImGui::GetFontSize(), 0.45f * viewport->WorkSize.x);
	ImGui::SetNextWindowPos(viewport->WorkPos);
	ImGui::SetNextWindowSize(ImVec2(sidebarWidth, viewport->WorkSize.y));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::Begin("##sidebar", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
										   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
	ImGui::PopStyleVar();
	const float full = -FLT_MIN;

	ImVec2 logoSize = recast::ui::logoSize();
	ImGui::PushFont(recast::ui::titleFont(), ImGui::GetStyle().FontSizeBase * 3.0f);
	const float titleHeight = ImGui::GetFontSize();
	if (logoSize.y > 0.0f) {
		float height = 1.15f * titleHeight;
		ImGui::Image(recast::ui::logo(), ImVec2(logoSize.x * height / logoSize.y, height));
		ImGui::SameLine();
		ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 0.5f * (height - titleHeight));
	}
	ImGui::TextColored(recast::ui::purple(), "ReCast");
	ImGui::PopFont();
	hint("Put any object into any photo.");

	// scene
	sectionHeading("Scene");
	ImGui::BeginDisabled(pythonJob.running());
	if (ImGui::Button("Choose scene photo...", ImVec2(full, 0.0f))) chooseScenePhoto();
	ImGui::Checkbox("Reuse saved scenes and objects", &useCache);
	if (ImGui::IsItemHovered()) ImGui::SetTooltip("A photo used before opens its saved scene, or reuses its object from the Objects tab");
	ImGui::EndDisabled();
	if (pythonJob.running()) {
		static const char* dots[] = { "", ".", "..", "..." };
		ImGui::TextColored(recast::ui::purple(), "%s%s", pythonLabel.c_str(), dots[static_cast<int>(ImGui::GetTime() * 3.0) % 4]);
	}
	else if (!pipelineStatus.empty()) {
		ImGui::TextWrapped("%s", pipelineStatus.c_str());
	}
	if (pythonFailed) drawPythonLog();

	// objects
	sectionHeading("Objects");
	ImGui::BeginDisabled(pythonJob.running());
	if (ImGui::Button("Add object from a photo...", ImVec2(full, 0.0f))) chooseObjectPhoto();
	ImGui::EndDisabled();
	drawObjectPanel();
	hint("Click an object to pick it. Drag to move it, scroll to rotate, Shift+scroll to resize. Add more from the Objects tab at the bottom.");

	if (sceneLoaded) {
		// lighting
		sectionHeading("Lighting");
		if (!lightingStatus.empty()) ImGui::TextWrapped("%s", lightingStatus.c_str());
		labelled("Exposure");
		float exposure = getExposure();
		if (ImGui::SliderFloat("##exposure", &exposure, -4.0f, 4.0f, "%.1f EV")) setExposure(exposure);
		ImGui::BeginDisabled(detectJob.valid());
		if (ImGui::Button("Detect lighting again", ImVec2(full, 0.0f))) detectLighting();
		ImGui::EndDisabled();
		if (openLightEditor) {
			showLights = true;
			ImGui::SetScrollHereY();
			openLightEditor = false;
		}
		ImGui::Checkbox("Show and edit lights", &showLights);
		if (showLights) {
			drawLightEditor();
			drawLightOverlay();
		}

		// room
		sectionHeading("Room");
		if (ImGui::Checkbox("Edit room", &editRoom) && !editRoom) {
			selectedProxy = -1;
			draggingProxy = false;
		}
		if (editRoom) {
			drawRoomEditor();
			drawRoomOverlay();
		}

		// save and export
		sectionHeading("Export");
		const float half = 0.5f * (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x);
		if (ImGui::Button(unsavedChanges ? "Save scene *" : "Save scene", ImVec2(half, 0.0f))) saveScene();
		ImGui::SameLine();
		if (recast::ui::accentButton("Export image", ImVec2(half, 0.0f))) exportRequested = true;

		// view and debug
		sectionHeading("View");
		ImGui::Checkbox("Show original photo", &showOriginal);
		if (supportsPathTracing()) {
			ImGui::Checkbox("Path tracing", &pathTrace);
			if (isPathTracing()) {
				ImGui::SameLine();
				ImGui::TextDisabled("%u samples", getPathTraceSamples());
				if (ImGui::Checkbox("Denoise", &denoise)) setDenoise(denoise);
			}
		}
		else {
			hint("Path tracing needs a GPU with ray tracing.");
		}
		ImGui::TextDisabled("%.0f fps, F12 saves a screenshot", ImGui::GetIO().Framerate);
	}

	ImGui::End();
}

// What Python printed, after a step failed
void Visualizer::drawPythonLog() {
	std::string log = pythonJob.output();
	ImGui::BeginChild("python log", ImVec2(0.0f, 9.0f * ImGui::GetFontSize()), ImGuiChildFlags_Borders,
					  ImGuiWindowFlags_HorizontalScrollbar);
	ImGui::TextUnformatted(log.c_str(), log.c_str() + log.size());
	ImGui::EndChild();
}

// A small card in the middle of the photo that asks for the next step: the scene photo at the start, the object photo
// once the scene is in, and how to move the object once it's added
void Visualizer::drawPrompt() {
	enum class Prompt { None, Working, Scene, Object, Placed };
	Prompt prompt = Prompt::None;
	if (pythonJob.running()) prompt = Prompt::Working;
	else if (!sceneLoaded) prompt = Prompt::Scene;
	else if (scene.objects.empty()) prompt = Prompt::Object;
	else if (showPlacedTip) prompt = Prompt::Placed;
	if (prompt == Prompt::None) return;

	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	const float em = ImGui::GetFontSize();
	const float width = std::min(21.0f * em, viewport->WorkSize.x - sidebarWidth - 2.0f * em);
	// centred in the space beside the sidebar and above the drawer
	ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + sidebarWidth + 0.5f * (viewport->WorkSize.x - sidebarWidth),
								   viewport->WorkPos.y + 0.5f * (viewport->WorkSize.y - drawerHeight)),
							ImGuiCond_Always, ImVec2(0.5f, 0.5f));
	ImGui::SetNextWindowSizeConstraints(ImVec2(width, 0.0f), ImVec2(width, FLT_MAX));
	ImGui::PushStyleColor(ImGuiCol_WindowBg, recast::ui::paperLight());
	ImGui::PushStyleColor(ImGuiCol_FrameBg, recast::ui::paper()); // fields stand out from the card
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(1.4f * em, 1.2f * em));
	ImGui::Begin("##prompt", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
										  ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
										  ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar);
	ImGui::PopStyleVar(2);

	auto title = [](const char* text) {
		ImGui::PushFont(recast::ui::boldFont(), ImGui::GetStyle().FontSizeBase * 1.5f);
		ImGui::TextColored(recast::ui::purple(), "%s", text);
		ImGui::PopFont();
	};
	auto text = [](const char* body) {
		ImGui::PushTextWrapPos(0.0f);
		ImGui::TextUnformatted(body);
		ImGui::PopTextWrapPos();
	};
	const ImVec2 full(-FLT_MIN, 2.2f * em);

	switch (prompt) {
	case Prompt::Working: {
		static const char* dots[] = { "", ".", "..", "..." };
		if (pendingObjectReload) {
			title("Select your object");
			text("A new window should pop up with your object photo. If you don't see it, check behind this window or "
				 "in the taskbar.");
			ImGui::Dummy(ImVec2(0.0f, 0.3f * em));
			text("In that window, drag a box around the object, then press Enter.");
			ImGui::Dummy(ImVec2(0.0f, 0.3f * em));
		}
		else {
			title("One moment");
		}
		ImGui::TextColored(recast::ui::purple(), "%s%s", pythonLabel.c_str(), dots[static_cast<int>(ImGui::GetTime() * 3.0) % 4]);
		ImGui::TextDisabled("This can take a minute the first time.");
		break;
	}
	case Prompt::Scene:
		{
			// as big as fits the card, centred
			const char* welcome = "welcome to ReCast!";
			const float base = ImGui::GetStyle().FontSizeBase;
			ImGui::PushFont(recast::ui::titleFont(), base * 3.0f);
			float fit = std::min(1.0f, ImGui::GetContentRegionAvail().x / ImGui::CalcTextSize(welcome).x);
			ImGui::PopFont();
			ImGui::PushFont(recast::ui::titleFont(), base * 3.0f * fit);
			ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 0.5f * (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(welcome).x));
			ImGui::TextColored(recast::ui::purple(), "%s", welcome);
			ImGui::PopFont();
		}
		ImGui::Dummy(ImVec2(0.0f, 0.3f * em));
		if (recast::ui::accentButton("Choose scene photo...", full)) chooseScenePhoto();
		if (pythonFailed) {
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.02f, 0.02f, 1.0f));
			text("That photo didn't work. Try another one, or see what went wrong below.");
			ImGui::PopStyleColor();
			drawPythonLog();
		}
		break;
	case Prompt::Object:
		title("Add an object");
		text("Choose a photo of the object to place. A new window will then pop up where you drag a box around the object.");
		ImGui::Dummy(ImVec2(0.0f, 0.3f * em));
		if (recast::ui::accentButton("Choose object photo...", full)) chooseObjectPhoto();
		if (!bank.empty()) hint("Or click one you made before in the Objects tab at the bottom, or drag it onto the photo.");
		if (pythonFailed) {
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.02f, 0.02f, 1.0f));
			text("That object didn't work. Try another photo, or see what went wrong below.");
			ImGui::PopStyleColor();
			drawPythonLog();
		}
		break;
	case Prompt::Placed:
		title("Your object is in!");
		text("Drag it to move it. Scroll to rotate it, Shift+scroll to resize it. When it looks right, export the image from the sidebar.");
		ImGui::Dummy(ImVec2(0.0f, 0.3f * em));
		if (recast::ui::accentButton("Got it", full)) showPlacedTip = false;
		break;
	default:
		break;
	}
	ImGui::End();
	ImGui::PopStyleColor(2);
}

std::string Visualizer::pythonCommand(const std::vector<std::string>& arguments) const {
	std::string joined;
	for (const std::string& argument : arguments) joined += " " + shellQuoted(argument);
#ifdef _WIN32
	std::string command = "cd /d " + shellQuoted(nativePath(RECAST_PYTHON_DIR)) + " && " + shellQuoted(pythonExecutable()) + " -u" + joined + " 2>&1";
	return shellQuoted(command); // cmd /c strips one pair of outer quotes
#else
	return "cd " + shellQuoted(RECAST_PYTHON_DIR) + " && " + shellQuoted(pythonExecutable()) + " -u" + joined + " 2>&1";
#endif
}

// The worker keeps torch and the models loaded, so steps skip ~25 s of start-up each.
// It warms up in the background while the photos are being picked
void Visualizer::startPythonWorker() {
	if (std::getenv("RECAST_NO_WORKER")) return;
	std::filesystem::path queue = std::filesystem::temp_directory_path() / ("recast_worker_" + std::to_string(recast::currentProcessId()));
	pythonWorker.start(pythonCommand({ "-m", "recast.worker", nativePath(queue), "--parent", std::to_string(recast::currentProcessId()) }), queue);
}

void Visualizer::runPython(const std::string& label, const std::string& module, const std::vector<std::string>& arguments,
						   const std::string& resultScene, bool reloadPhoto, bool reloadObject) {
	if (pythonJob.running()) return;
	// Python rewrites scene.json and the reload replaces what's in memory, so unsaved drags are saved first
	if (sceneLoaded && unsavedChanges) saveScene();

	pythonLabel = label;
	pendingScenePath = resultScene;
	pendingPhotoReload = reloadPhoto;
	pendingObjectReload = reloadObject;
	pythonFailed = false;
	loadIfFollowUpFails = false;
	pipelineStatus = label + "...";
	if (pythonWorker.alive()) {
		nlohmann::json request = { { "module", module }, { "args", arguments } };
		pythonJob.startInWorker(pythonWorker.queue(), request.dump(), [this]() { return pythonWorker.alive(); });
	}
	else {
		std::vector<std::string> command = { "-m", module };
		command.insert(command.end(), arguments.begin(), arguments.end());
		pythonJob.start(pythonCommand(command));
	}
}

void Visualizer::finishPython() {
	if (!pythonJob.finished()) return;

	bool succeeded = pythonJob.exitCode() == 0;
	pythonFailed = !succeeded;
	pipelineStatus = pythonLabel + (succeeded ? " done" : " failed");

	// the next step runs before loading, so lighting is detected once, with the whole room
	if (succeeded && !followUpArguments.empty()) {
		std::string label = std::move(followUpLabel);
		std::vector<std::string> arguments = std::move(followUpArguments);
		followUpLabel.clear();
		followUpArguments.clear();
		std::string result = pendingScenePath;
		bool reloadPhoto = pendingPhotoReload, reloadObject = pendingObjectReload;
		runPython(label, "recast", arguments, result, reloadPhoto, reloadObject);
		loadIfFollowUpFails = true;
		return;
	}
	followUpLabel.clear();
	followUpArguments.clear();
	bool load = succeeded || loadIfFollowUpFails;
	loadIfFollowUpFails = false;
	if (!load || pendingScenePath.empty()) {
		pendingScenePath.clear();
		return;
	}

	// Same file names get rewritten (photo.jpg, object.glb), so the loaded copies are dropped to force a reload
	if (pendingPhotoReload) loadedPhotoPath.clear();
	if (pendingObjectReload) {
		// object.glb may have been rewritten under the same name: load every object file again
		for (auto& [path, mesh] : objectMeshes) {
			if (mesh) destroyMesh(mesh);
		}
		objectMeshes.clear();
		objectMeshOf.clear();
		selectNewestObject = true;
		if (succeeded) showPlacedTip = true;
	}
	scenePath = pendingScenePath;
	pendingScenePath.clear();
	loadScene();
	sceneWatcher = foto::FileWatcher(scenePath);
}

void Visualizer::chooseScenePhoto() {
	std::string photo = pickPhoto("scene photo");
	if (photo.empty()) return;
	// the folder is named after the photo's bytes too: the same photo opens the scene built (and edited) before
	std::string name = folderName(photo) + "_" + contentHash(photo);
	std::filesystem::path folder = std::filesystem::path(RECAST_DATA_DIR) / name;
	std::filesystem::path result = folder / "scene.json";
	std::error_code error;
	if (useCache && std::filesystem::exists(result, error) && std::filesystem::exists(folder / "detect.json", error)) {
		if (sceneLoaded && unsavedChanges) saveScene();
		scenePath = nativePath(result);
		loadedPhotoPath.clear();
		openCachedScene = true;
		pipelineStatus = "Opened the scene saved for this photo (untick reuse to build it again)";
		return;
	}
	std::string dataRoot = nativePath(RECAST_DATA_DIR);
	runPython("building scene " + name, "recast", { "scene", photo, "--name", name, "--data-root", dataRoot, "--overwrite" },
			  nativePath(result), true, false);
	followUpLabel = "finding walls and furniture";
	followUpArguments = { "detect", "--name", name, "--data-root", dataRoot };
}

// Made once into the bank (data/objects) and added to this room as one more object. It starts at a typical tabletop
// size; the size slider and Shift+scroll fit it to the room
void Visualizer::chooseObjectPhoto() {
	std::string photo = pickPhoto("object photo");
	if (photo.empty()) return;
	std::vector<std::string> arguments = { photo, "--mode", "generated", "--height", "0.30",
										   "--out", nativePath(scene.directory),
										   "--bank", nativePath(std::filesystem::path(RECAST_DATA_DIR) / "objects") };
	if (!useCache) arguments.push_back("--rebuild");
	runPython("adding object", "recast.object", arguments, scenePath, false, true);
}

// Boxes share one unit cube placed by a matrix, so dragging and resizing never rebuild a mesh.
// Planes keep the mesh they were built with and move by the offset from where it was built
void Visualizer::rebuildProxyMeshes() {
	for (foto::Mesh* mesh : proxyMeshes) {
		if (mesh && mesh != unitBoxMesh) destroyMesh(mesh);
	}
	proxyMeshes.clear();
	proxyAlbedoMaterials.clear();
	proxyMeshOrigins.clear();
	for (const recast::Proxy& proxy : scene.proxies) {
		foto::Mesh* mesh = nullptr;
		if (proxy.type == "box") mesh = unitBoxMesh;
		else {
			recast::MeshData data = recast::buildProxyMesh(proxy);
			if (!data.indices.empty()) mesh = createMesh(data.vertices, data.indices);
		}
		proxyMeshes.push_back(mesh);
		proxyMeshOrigins.push_back(proxy.point);
		const recast::Vec3f& albedo = proxy.material.albedo;
		auto mask = proxyMaskIds.find(proxy.id);
		proxyAlbedoMaterials.push_back(createMaterial({ .baseColor = { albedo[0], albedo[1], albedo[2], 1.0f },
														.maskId = mask == proxyMaskIds.end() ? 0u : mask->second }));
	}
}

Mat4 Visualizer::proxyMatrix(size_t index) const {
	const recast::Proxy& proxy = scene.proxies[index];
	if (proxy.type == "box") {
		Vec3 size(std::max(proxy.max[0] - proxy.min[0], 1e-3f), std::max(proxy.max[1] - proxy.min[1], 1e-3f),
				  std::max(proxy.max[2] - proxy.min[2], 1e-3f));
		return Mat4::translationM(proxy.min[0], proxy.min[1], proxy.min[2]) * Mat4::scaleM(size.x, size.y, size.z);
	}
	const recast::Vec3f& built = proxyMeshOrigins[index];
	return Mat4::translationM(proxy.point[0] - built[0], proxy.point[1] - built[1], proxy.point[2] - built[2]);
}

namespace {
	bool isWall(const recast::Proxy& proxy) {
		return proxy.type == "plane" && std::abs(proxy.normal[1]) < 0.5f;
	}

	// Closest point on the line p + u*s to the ray o + d*t: returns s
	float closestOnLine(const Vec3& p, const Vec3& u, const Vec3& o, const Vec3& d) {
		Vec3 w = p - o;
		float b = u.dot(d), denom = 1.0f - b * b;
		if (std::abs(denom) < 1e-6f) return 0.0f;
		return (b * d.dot(w) - u.dot(w)) / denom;
	}

	constexpr ImU32 EDGE_COLOR = IM_COL32(255, 255, 255, 170);
	constexpr ImU32 HOVER_COLOR = IM_COL32(255, 230, 90, 255);
	constexpr ImU32 SELECTED_COLOR = IM_COL32(255, 150, 40, 255);
	constexpr ImU32 WALL_COLOR = IM_COL32(110, 200, 255, 200);
	constexpr float HANDLE_RADIUS = 7.0f;
}

// From detect.json: names ("chair", "table"...) for the room list and labels, and the owner image: each detected
// object's cutout gets a number (its boxes' Material::maskId), which the path tracer uses to look past a box
// wherever the photo shows something other than that object
void Visualizer::loadProxyLabels() {
	proxyLabels.clear();
	proxyMaskIds.clear();
	ownerTexture = 0;
	ownerPixels.clear();
	ownerWidth = ownerHeight = 0;
	std::ifstream file(scene.directory / "detect.json");
	if (!file) return;
	try {
		nlohmann::json detect = nlohmann::json::parse(file);
		std::vector<unsigned char>& owners = ownerPixels;
		foto::u32 next = 1;
		for (const auto& object : detect.value("objects", nlohmann::json::array())) {
			std::string label = object.value("label", "object");
			const auto& ids = object.value("proxy_ids", nlohmann::json::array());
			bool seatBack = object.value("shape", ids.size() == 2 ? "seat_back" : "box") == "seat_back";
			for (size_t k = 0; k < ids.size(); k++) {
				proxyLabels[ids[k].get<foto::u32>()] = seatBack ? label + (k == 0 ? " (seat)" : " (back)") : label;
			}

			std::string maskPath = object.value("mask", "");
			if (maskPath.empty() || next > 255) continue;
			int width = 0, height = 0, channels = 0;
			unsigned char* mask = stbi_load((scene.directory / maskPath).string().c_str(), &width, &height, &channels, 1);
			if (!mask) continue;
			if (owners.empty()) {
				ownerWidth = width;
				ownerHeight = height;
				owners.assign(static_cast<size_t>(width) * height * 4, 0);
			}
			if (width == ownerWidth && height == ownerHeight) {
				for (size_t i = 0; i < static_cast<size_t>(width) * height; i++) {
					if (mask[i] > 127) owners[i * 4] = static_cast<unsigned char>(next);
				}
				for (const auto& id : ids) proxyMaskIds[id.get<foto::u32>()] = next;
				next++;
			}
			stbi_image_free(mask);
		}
		if (!owners.empty()) ownerTexture = createDataTexture(owners.data(), ownerWidth, ownerHeight);
	}
	catch (const std::exception& e) {
		std::cerr << "[recast] detect.json: " << e.what() << std::endl;
	}
}

std::string Visualizer::proxyName(size_t index) const {
	const recast::Proxy& proxy = scene.proxies[index];
	auto label = proxyLabels.find(proxy.id);
	if (label != proxyLabels.end()) return label->second;
	if (proxy.type == "box") return "box";
	if (proxy.type == "hull") return "object";
	if (isWall(proxy)) return "wall";
	return proxy.normal[1] > 0.0f ? "floor" : "ceiling";
}

// World point -> ImGui screen position through the photo's camera. False when it's behind the camera
bool Visualizer::projectToScreen(const Vec3& point, ImVec2& out) {
	const recast::SceneCamera& c = scene.camera;
	Vec3 d = point - Vec3(c.position[0], c.position[1], c.position[2]);
	Vec3 local;
	for (int j = 0; j < 3; j++) local[j] = c.rotation[0][j] * d.x + c.rotation[1][j] * d.y + c.rotation[2][j] * d.z;
	if (local.z > -0.05f) return false;
	float u = c.cx + c.fx * local.x / -local.z;
	float v = c.cy - c.fy * local.y / -local.z;
	enginemath::Vec2 offset = getViewportOffset(), size = getViewportSize();
	ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
	out = ImVec2((offset.x + u / std::max(1.0f, static_cast<float>(scene.image_size[0])) * size.x) / scale.x,
				 (offset.y + v / std::max(1.0f, static_cast<float>(scene.image_size[1])) * size.y) / scale.y);
	return true;
}

void Visualizer::drawSegment(ImDrawList* list, Vec3 a, Vec3 b, ImU32 color, float thickness) {
	// clip against a plane just in front of the camera, so lines that pass behind it still draw
	const recast::SceneCamera& c = scene.camera;
	Vec3 eye(c.position[0], c.position[1], c.position[2]);
	Vec3 forward(-c.rotation[0][2], -c.rotation[1][2], -c.rotation[2][2]);
	float da = (a - eye).dot(forward) - 0.06f, db = (b - eye).dot(forward) - 0.06f;
	if (da < 0.0f && db < 0.0f) return;
	if (da < 0.0f) a = a + (b - a) * (da / (da - db));
	if (db < 0.0f) b = b + (a - b) * (db / (db - da));
	ImVec2 sa, sb;
	if (projectToScreen(a, sa) && projectToScreen(b, sb)) list->AddLine(sa, sb, color, thickness);
}

// Face centres of the selected box: x-, x+, y-, y+, z-, z+
std::array<Vec3, 6> Visualizer::boxHandles(const recast::Proxy& box) const {
	Vec3 lo(box.min[0], box.min[1], box.min[2]), hi(box.max[0], box.max[1], box.max[2]), mid = (lo + hi) * 0.5f;
	std::array<Vec3, 6> handles;
	for (int a = 0; a < 3; a++) {
		handles[a * 2] = mid;
		handles[a * 2][a] = lo[a];
		handles[a * 2 + 1] = mid;
		handles[a * 2 + 1][a] = hi[a];
	}
	return handles;
}

// Where a wall's handle sits: on the floor, straight ahead of the camera
bool Visualizer::wallHandle(const recast::Proxy& wall, Vec3& handle) const {
	const recast::SceneCamera& c = scene.camera;
	Vec3 eye(c.position[0], c.position[1], c.position[2]);
	Vec3 forward(-c.rotation[0][2], 0.0f, -c.rotation[2][2]);
	Vec3 normal(wall.normal[0], wall.normal[1], wall.normal[2]), point(wall.point[0], wall.point[1], wall.point[2]);
	float facing = forward.dot(normal);
	if (std::abs(facing) < 1e-3f) return false;
	float t = (point - eye).dot(normal) / facing;
	if (t <= 0.0f) return false;
	handle = eye + forward * t;
	handle.y = 0.0f;
	return true;
}

int Visualizer::boxUnderCursor(const Vec3& origin, const Vec3& direction) const {
	int best = -1;
	float bestT = 1e30f;
	for (size_t i = 0; i < scene.proxies.size(); i++) {
		const recast::Proxy& proxy = scene.proxies[i];
		float t = 0.0f;
		Vec3 normal;
		bool hit = false;
		if (proxy.type == "box") {
			hit = recast::rayAabb(origin, direction, Vec3(proxy.min[0], proxy.min[1], proxy.min[2]), Vec3(proxy.max[0], proxy.max[1], proxy.max[2]), t);
		}
		else if (proxy.type == "hull") {
			hit = recast::rayHull(proxy, origin, direction, t, normal);
		}
		if (hit && t < bestT) {
			bestT = t;
			best = static_cast<int>(i);
		}
	}
	return best;
}

// Boxes of one detected object (its outline slices, or a seat and back) move, select and delete together
bool Visualizer::sameGroup(int a, int b) const {
	if (a < 0 || b < 0) return false;
	if (a == b) return true;
	auto maskA = proxyMaskIds.find(scene.proxies[a].id), maskB = proxyMaskIds.find(scene.proxies[b].id);
	return maskA != proxyMaskIds.end() && maskB != proxyMaskIds.end() && maskA->second == maskB->second;
}

int Visualizer::groupSize(int index) const {
	int count = 0;
	for (size_t i = 0; i < scene.proxies.size(); i++) count += sameGroup(index, static_cast<int>(i)) ? 1 : 0;
	return count;
}

// Click a box to select it and drag to slide it across the floor; drag the dots on the selected box to
// resize it, or the dot on a selected wall (pick walls in the list) to move the wall in or out
bool Visualizer::updateRoomEditing() {
	bool leftDown = getWindow().isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
	bool pressed = leftDown && !roomMouseWasDown && !ImGui::GetIO().WantCaptureMouse;
	roomMouseWasDown = leftDown;
	Vec3 origin, direction;
	bool haveRay = getCursorRay(origin, direction);
	ImVec2 mouse = ImGui::GetIO().MousePos;
	auto nearMouse = [&](const Vec3& point) {
		ImVec2 screen;
		if (!projectToScreen(point, screen)) return false;
		float dx = screen.x - mouse.x, dy = screen.y - mouse.y;
		return dx * dx + dy * dy <= (HANDLE_RADIUS + 4.0f) * (HANDLE_RADIUS + 4.0f);
	};
	bool haveSelection = selectedProxy >= 0 && selectedProxy < static_cast<int>(scene.proxies.size());

	hoveredHandle = -1;
	if (haveSelection && !draggingProxy && groupSize(selectedProxy) == 1 && scene.proxies[selectedProxy].type != "hull") {
		const recast::Proxy& selected = scene.proxies[selectedProxy];
		Vec3 handle;
		if (selected.type == "box") {
			std::array<Vec3, 6> handles = boxHandles(selected);
			for (int h = 0; h < 6 && hoveredHandle < 0; h++) {
				if (nearMouse(handles[h])) hoveredHandle = h;
			}
		}
		else if (isWall(selected) && wallHandle(selected, handle) && nearMouse(handle)) hoveredHandle = 6;
	}
	if (!draggingProxy) hoveredProxy = haveRay ? boxUnderCursor(origin, direction) : -1;

	if (pressed && haveRay) {
		if (hoveredHandle >= 0) {
			draggingProxy = true;
			draggingHandle = hoveredHandle;
			proxyAtGrab = scene.proxies[selectedProxy];
			if (hoveredHandle == 6) wallHandle(proxyAtGrab, proxyGrab);
			else proxyGrab = boxHandles(proxyAtGrab)[hoveredHandle];
		}
		else if (hoveredProxy >= 0) {
			selectedProxy = hoveredProxy;
			draggingProxy = true;
			draggingHandle = -1;
			proxyAtGrab = scene.proxies[selectedProxy];
			roomAtGrab = scene.proxies;
			float t = 0.0f;
			recast::rayAabb(origin, direction, Vec3(proxyAtGrab.min[0], proxyAtGrab.min[1], proxyAtGrab.min[2]),
							Vec3(proxyAtGrab.max[0], proxyAtGrab.max[1], proxyAtGrab.max[2]), t);
			proxyGrab = origin + direction * t;
		}
		else {
			selectedProxy = -1;
		}
	}
	if (!leftDown) draggingProxy = false;
	if (!draggingProxy || !haveRay || selectedProxy < 0) return draggingProxy;

	recast::Proxy& proxy = scene.proxies[selectedProxy];
	if (draggingHandle == 6) {
		Vec3 normal(proxy.normal[0], proxy.normal[1], proxy.normal[2]);
		float s = closestOnLine(proxyGrab, normal, origin, direction);
		for (int a = 0; a < 3; a++) proxy.point[a] = proxyAtGrab.point[a] + normal[a] * s;
	}
	else if (draggingHandle >= 0) {
		int axis = draggingHandle / 2;
		bool high = draggingHandle % 2 == 1;
		Vec3 along(0.0f);
		along[axis] = 1.0f;
		float value = proxyGrab[axis] + closestOnLine(proxyGrab, along, origin, direction);
		if (high) proxy.max[axis] = std::max(value, proxy.min[axis] + 0.02f);
		else proxy.min[axis] = std::min(axis == 1 ? std::max(value, 0.0f) : value, proxy.max[axis] - 0.02f);
	}
	else if (std::abs(direction.y) > 1e-4f) {
		float t = (proxyGrab.y - origin.y) / direction.y;
		if (t > 0.0f) {
			Vec3 delta = origin + direction * t - proxyGrab;
			for (size_t i = 0; i < scene.proxies.size() && i < roomAtGrab.size(); i++) {
				if (!sameGroup(selectedProxy, static_cast<int>(i))) continue;
				for (int a : { 0, 2 }) {
					scene.proxies[i].min[a] = roomAtGrab[i].min[a] + delta[a];
					scene.proxies[i].max[a] = roomAtGrab[i].max[a] + delta[a];
					scene.proxies[i].point[a] = roomAtGrab[i].point[a] + delta[a]; // a hull moves by its anchor
				}
			}
		}
	}
	unsavedChanges = true;
	return true;
}

// Outlines, names and resize dots drawn over the photo (nothing is filled, so the photo stays readable)
void Visualizer::drawRoomOverlay() {
	ImDrawList* list = ImGui::GetForegroundDrawList();
	for (size_t i = 0; i < scene.proxies.size(); i++) {
		const recast::Proxy& proxy = scene.proxies[i];
		bool selected = sameGroup(static_cast<int>(i), selectedProxy), hovered = sameGroup(static_cast<int>(i), hoveredProxy);
		bool firstOfGroup = true;
		for (size_t j = 0; j < i; j++) firstOfGroup &= !sameGroup(static_cast<int>(i), static_cast<int>(j));
		if (proxy.type == "box") {
			ImU32 color = selected ? SELECTED_COLOR : hovered ? HOVER_COLOR : EDGE_COLOR;
			float thickness = selected ? 2.5f : hovered ? 2.0f : 1.2f;
			Vec3 lo(proxy.min[0], proxy.min[1], proxy.min[2]), hi(proxy.max[0], proxy.max[1], proxy.max[2]);
			auto corner = [&](int k) { return Vec3((k & 1) ? hi.x : lo.x, (k & 2) ? hi.y : lo.y, (k & 4) ? hi.z : lo.z); };
			for (int k = 0; k < 8; k++) {
				for (int bit : { 1, 2, 4 }) {
					if (!(k & bit)) drawSegment(list, corner(k), corner(k | bit), color, thickness);
				}
			}
			ImVec2 at;
			if (firstOfGroup && projectToScreen(Vec3((lo.x + hi.x) * 0.5f, hi.y, (lo.z + hi.z) * 0.5f), at)) {
				std::string name = proxyName(i);
				ImVec2 size = ImGui::CalcTextSize(name.c_str());
				ImVec2 corner0(at.x - size.x * 0.5f - 3.0f, at.y - size.y - 8.0f);
				list->AddRectFilled(corner0, ImVec2(corner0.x + size.x + 6.0f, corner0.y + size.y + 2.0f), IM_COL32(0, 0, 0, 150), 3.0f);
				list->AddText(ImVec2(corner0.x + 3.0f, corner0.y + 1.0f), color, name.c_str());
			}
			if (static_cast<int>(i) == selectedProxy && groupSize(selectedProxy) == 1 && proxy.type == "box") {
				std::array<Vec3, 6> handles = boxHandles(proxy);
				for (int h = 0; h < 6; h++) {
					ImVec2 screen;
					if (!projectToScreen(handles[h], screen)) continue;
					bool active = h == hoveredHandle || (draggingProxy && h == draggingHandle);
					list->AddCircleFilled(screen, active ? HANDLE_RADIUS + 2.0f : HANDLE_RADIUS, active ? HOVER_COLOR : SELECTED_COLOR);
					list->AddCircle(screen, active ? HANDLE_RADIUS + 2.0f : HANDLE_RADIUS, IM_COL32(0, 0, 0, 200), 0, 1.5f);
				}
			}
		}
		else if (proxy.type == "hull") {
			ImU32 color = selected ? SELECTED_COLOR : hovered ? HOVER_COLOR : EDGE_COLOR;
			float thickness = selected ? 2.0f : hovered ? 1.6f : 1.0f;
			Vec3 anchor(proxy.point[0], proxy.point[1], proxy.point[2]);
			auto vertex = [&](foto::u32 k) { return anchor + Vec3(proxy.vertices[k][0], proxy.vertices[k][1], proxy.vertices[k][2]); };
			for (const auto& face : proxy.faces) {
				for (int e = 0; e < 3; e++) {
					foto::u32 a = face[e], b = face[(e + 1) % 3];
					if (a < b && a < proxy.vertices.size() && b < proxy.vertices.size()) drawSegment(list, vertex(a), vertex(b), color, thickness);
				}
			}
			ImVec2 at;
			if (firstOfGroup && projectToScreen(Vec3(0.5f * (proxy.min[0] + proxy.max[0]), proxy.max[1], 0.5f * (proxy.min[2] + proxy.max[2])), at)) {
				std::string name = proxyName(i);
				ImVec2 size = ImGui::CalcTextSize(name.c_str());
				ImVec2 corner0(at.x - size.x * 0.5f - 3.0f, at.y - size.y - 8.0f);
				list->AddRectFilled(corner0, ImVec2(corner0.x + size.x + 6.0f, corner0.y + size.y + 2.0f), IM_COL32(0, 0, 0, 150), 3.0f);
				list->AddText(ImVec2(corner0.x + 3.0f, corner0.y + 1.0f), color, name.c_str());
			}
		}
		else if (isWall(proxy)) {
			// the wall's line on the floor, far enough either side to cross the photo
			Vec3 normal(proxy.normal[0], 0.0f, proxy.normal[2]), point(proxy.point[0], 0.0f, proxy.point[2]);
			Vec3 tangent(-normal.z, 0.0f, normal.x);
			ImU32 color = selected ? SELECTED_COLOR : WALL_COLOR;
			for (float height : { 0.0f, 2.5f }) {
				Vec3 up(0.0f, height, 0.0f);
				drawSegment(list, point + up - tangent * 30.0f, point + up + tangent * 30.0f, color, selected ? 2.5f : 1.2f);
			}
			Vec3 handle;
			if (selected && wallHandle(proxy, handle)) {
				ImVec2 screen;
				if (projectToScreen(handle, screen)) {
					bool active = hoveredHandle == 6 || (draggingProxy && draggingHandle == 6);
					list->AddCircleFilled(screen, active ? HANDLE_RADIUS + 2.0f : HANDLE_RADIUS, active ? HOVER_COLOR : SELECTED_COLOR);
				}
			}
		}
	}
}

void Visualizer::drawRoomEditor() {
	hint("Click a piece to select it. Drag to move it, drag its dots to resize it.");
	if (ImGui::BeginListBox("##room", ImVec2(-FLT_MIN, 6.5f * ImGui::GetTextLineHeightWithSpacing()))) {
		for (size_t i = 0; i < scene.proxies.size(); i++) {
			const recast::Proxy& proxy = scene.proxies[i];
			if (proxy.type != "box" && proxy.type != "hull" && !isWall(proxy)) continue;
			bool firstOfGroup = true;
			for (size_t j = 0; j < i; j++) firstOfGroup &= !sameGroup(static_cast<int>(i), static_cast<int>(j));
			if (!firstOfGroup) continue;
			std::string label = proxyName(i) + "##" + std::to_string(i);
			if (ImGui::Selectable(label.c_str(), sameGroup(static_cast<int>(i), selectedProxy))) selectedProxy = static_cast<int>(i);
			if (ImGui::IsItemHovered()) hoveredProxy = static_cast<int>(i);
		}
		ImGui::EndListBox();
	}
	const float third = splitWidth(3);
	if (ImGui::Button("Add box", ImVec2(third, 0.0f))) {
		recast::Proxy box;
		box.type = "box";
		box.min = { shadowCenter.x - 0.25f, 0.0f, shadowCenter.z - 0.25f };
		box.max = { shadowCenter.x + 0.25f, 0.5f, shadowCenter.z + 0.25f };
		for (const recast::Proxy& proxy : scene.proxies) box.id = std::max(box.id, proxy.id + 1);
		scene.proxies.push_back(box);
		selectedProxy = static_cast<int>(scene.proxies.size()) - 1;
		proxyMeshesDirty = true; // rebuilt before the next frame's draws: this frame's are already queued
		unsavedChanges = true;
	}
	ImGui::SameLine();
	bool haveSelection = selectedProxy >= 0 && selectedProxy < static_cast<int>(scene.proxies.size());
	ImGui::BeginDisabled(!haveSelection);
	if (ImGui::Button("Delete", ImVec2(third, 0.0f)) && haveSelection) {
		std::vector<recast::Proxy> kept;
		for (size_t i = 0; i < scene.proxies.size(); i++) {
			if (!sameGroup(selectedProxy, static_cast<int>(i))) kept.push_back(scene.proxies[i]);
		}
		scene.proxies = kept;
		selectedProxy = -1;
		proxyMeshesDirty = true;
		unsavedChanges = true;
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button("Reset room", ImVec2(third, 0.0f))) {
		scene.proxies = detectedProxies;
		selectedProxy = -1;
		proxyMeshesDirty = true;
		unsavedChanges = true;
	}
}
