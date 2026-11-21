#include "visualizer.hpp"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <engine/platform/window.hpp>
#include <engine/rhi_vulkan/vkPipeline.hpp>

#include "scene/proxyMesh.hpp"
#include "scene/raycast.hpp"
#include "scene/placement.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cctype>
#include <fstream>
#include <map>
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
	constexpr const char* OBJECT_MODES[] = { "generated", "generated", "card" };
	constexpr const char* OBJECT_SHAPES[] = { "3D model", "glass", "flat" };
	constexpr int GLASS_SHAPE = 1;

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

	recast::Proxy markerBox;
	markerBox.type = "box";
	markerBox.min = { -0.5f, -0.5f, -0.5f };
	markerBox.max = { 0.5f, 0.5f, 0.5f };
	recast::MeshData markerData = recast::buildProxyMesh(markerBox);
	lightMarker = createMesh(markerData.vertices, markerData.indices);

	recast::Proxy unitBox;
	unitBox.type = "box";
	unitBox.min = { 0.0f, 0.0f, 0.0f };
	unitBox.max = { 1.0f, 1.0f, 1.0f };
	recast::MeshData unitBoxData = recast::buildProxyMesh(unitBox);
	unitBoxMesh = createMesh(unitBoxData.vertices, unitBoxData.indices);
	selectedProxyMaterial = createMaterial({ .baseColor = { 1.0f, 0.55f, 0.1f, 0.35f } });
	lightMarkerMaterial = createMaterial({ .baseColor = { 0.0f, 0.0f, 0.0f, 1.0f }, .emissive = { 4.0f, 3.6f, 1.2f, 0.0f } });
	selectedMarkerMaterial = createMaterial({ .baseColor = { 0.0f, 0.0f, 0.0f, 1.0f }, .emissive = { 0.8f, 3.0f, 4.0f, 0.0f } });

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

	std::string objectPath = loaded.object.path.empty() ? "" : loaded.resolve(loaded.object.path);
	if (objectPath != loadedObjectPath) {
		if (objectMesh) destroyMesh(objectMesh);
		objectMesh = nullptr;
		loadedObjectPath.clear();
		if (!objectPath.empty()) {
			try {
				objectMesh = loadModel(objectPath);
				loadedObjectPath = objectPath;
			}
			catch (const std::exception& e) {
				std::cerr << "[recast] " << e.what() << std::endl;
			}
		}
	}

	scene = std::move(loaded);
	sceneLoaded = true;
	detectedProxies = scene.proxies;
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
	if (recast::fitLightsFromScene(scene).empty()) detectLighting();
	else lightingStatus = std::to_string(scene.lights.size()) + " light(s) from scene.json";

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

	lightingStatus = "detecting lighting...";
	detectStarted = std::chrono::steady_clock::now();
	detectGaveUp = false;
	recast::SceneFile sceneCopy = scene;
	recast::LinearImage photo = photoPixels;
	int stride = lightingStride(8);
	detectJob = std::async(std::launch::async, [sceneCopy, photo, stride]() {
		DetectJob job;
		try {
			std::vector<recast::LightingSample> samples = recast::collectLightingSamples(sceneCopy, photo, stride);
			job.fit = recast::autoFitLighting(sceneCopy, photo, samples, job.lights);
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
	lightingStatus = "no light source detected (" + reason + "): add lights by hand with debug > show and edit lights > add light";
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
		lightingStatus = "lighting detection failed: " + job.error;
		return;
	}

	scene.lights.clear();
	int directionalCount = 0;
	for (size_t i = 0; i < job.lights.size(); i++) {
		const recast::FitLight& fitted = job.lights[i];
		recast::SceneLight light;
		light.type = fitted.directional ? "directional" : "point";
		light.position = { fitted.position.x, fitted.position.y, fitted.position.z };
		light.direction = { fitted.direction.x, fitted.direction.y, fitted.direction.z };
		light.intensity = { job.fit.intensities[i].x, job.fit.intensities[i].y, job.fit.intensities[i].z };
		if (fitted.directional) light.angular_radius_deg = fitted.size * 180.0f / enginemath::PI;
		else light.radius = fitted.size;
		scene.lights.push_back(light);
		directionalCount += fitted.directional ? 1 : 0;
	}
	scene.lighting.ambient = { job.fit.ambient.x, job.fit.ambient.y, job.fit.ambient.z };
	detectedLights = scene.lights;
	detectedAmbient = scene.lighting.ambient;
	selectedLight = -1;
	unsavedChanges = true;

	int seconds = static_cast<int>(std::chrono::duration<double>(std::chrono::steady_clock::now() - detectStarted).count());
	lightingStatus = "detected " + std::to_string(job.lights.size() - directionalCount) + " point + " +
					 std::to_string(directionalCount) + " directional light(s) in " + std::to_string(seconds) + " s";
	std::cout << "[recast] " << lightingStatus << std::endl;
	if (job.lights.empty()) askForManualLights("none found in the photo");
}

void Visualizer::onUpdate(foto::f32 deltaTime) {
	if (sceneWatcher.changed()) loadScene();

	bool screenshotKeyDown = getWindow().isKeyDown(GLFW_KEY_F12);
	if (screenshotKeyDown && !screenshotKeyWasDown) screenshot("screenshot.png");
	screenshotKeyWasDown = screenshotKeyDown;

	finishPython();
	if (!sceneLoaded) return;
	if (proxyMeshesDirty) {
		proxyMeshesDirty = false;
		rebuildProxyMeshes();
	}

	applySceneCamera();
	finishLightingDetection();
	applySceneLighting();
	if (editRoom) updateRoomEditing();
	else if (!(showLights && updateLightEditing())) updateObjectInteraction();
	setPathTracing(pathTrace && !showOriginal && !editRoom && !showLights && supportsPathTracing(), photoMaterial);
	float pixelsPerPhotoPixel = getViewportSize().x / std::max(static_cast<float>(scene.image_size[0]), 1.0f);
	setCameraMatch(cameraMatch.grain, cameraMatch.blur * pixelsPerPhotoPixel);
	drawScene();
	if (exportRequested) exportRender();
}

Mat4 Visualizer::objectTransform() const {
	return recast::objectMatrix(scene.object.transform);
}

// Where the object goes for this mouse ray: the first surface it points at, or the floor in front of it when that's a
// wall or the side of a box, so the object never ends up hidden behind something
recast::RayHit Visualizer::supportUnderCursor(const Vec3& origin, const Vec3& direction) {
	constexpr float supportNormalY = 0.7f;
	recast::RayHit hit = recast::raycastProxies(scene.proxies, origin, direction);
	if (!hit.valid() || hit.normal.y >= supportNormalY) return hit;
	return recast::raycastProxies(scene.proxies, hit.point + hit.normal * 0.02f, Vec3(0.0f, -1.0f, 0.0f), supportNormalY);
}

// Light markers sit at point lights, and 2 m back along a directional light from the middle of the room
Vec3 Visualizer::markerPosition(const recast::SceneLight& light) const {
	if (light.type == "directional") return shadowCenter - Vec3(light.direction[0], light.direction[1], light.direction[2]).normalized() * 2.0f;
	return Vec3(light.position[0], light.position[1], light.position[2]);
}

// Click a light marker to select it, drag to move it in the plane facing the camera. True while a light is being dragged
bool Visualizer::updateLightEditing() {
	bool leftDown = getWindow().isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
	bool pressed = leftDown && !lightMouseWasDown;
	lightMouseWasDown = leftDown;
	Vec3 origin, direction;
	bool haveRay = getCursorRay(origin, direction);

	if (pressed && haveRay) {
		constexpr float pickRadius = 0.12f;
		float closest = 1e30f;
		int picked = -1;
		for (size_t i = 0; i < scene.lights.size(); i++) {
			if (scene.lights[i].type != "point" && scene.lights[i].type != "directional") continue;
			Vec3 toMarker = markerPosition(scene.lights[i]) - origin;
			float t = toMarker.dot(direction);
			if (t <= 0.0f || toMarker.dot(toMarker) - t * t > pickRadius * pickRadius || t >= closest) continue;
			closest = t;
			picked = static_cast<int>(i);
		}
		if (picked >= 0) {
			selectedLight = picked;
			draggingLight = true;
			lightPlanePoint = markerPosition(scene.lights[picked]);
			lightPlaneNormal = direction;
		}
	}
	if (!leftDown) draggingLight = false;

	if (draggingLight && haveRay && selectedLight >= 0 && selectedLight < static_cast<int>(scene.lights.size())) {
		float facing = direction.dot(lightPlaneNormal);
		float t = std::abs(facing) > 1e-4f ? (lightPlanePoint - origin).dot(lightPlaneNormal) / facing : -1.0f;
		if (t > 0.0f) {
			Vec3 at = origin + direction * t;
			recast::SceneLight& light = scene.lights[selectedLight];
			if (light.type == "directional") {
				Vec3 toCenter = shadowCenter - at;
				if (toCenter.dot(toCenter) > 1e-6f) {
					toCenter.normalize();
					light.direction = { toCenter.x, toCenter.y, toCenter.z };
				}
			}
			else {
				light.position = { at.x, at.y, at.z };
			}
			unsavedChanges = true;
		}
	}
	return draggingLight;
}

// A point light 1.2 m above the object, as strong as the first point light (or a neutral one)
void Visualizer::addLight() {
	recast::SceneLight light;
	light.type = "point";
	const recast::Vec3f& base = scene.object.transform.position;
	Vec3 above = objectMesh ? Vec3(base[0], base[1] + 1.2f, base[2]) : shadowCenter + Vec3(0.0f, 1.2f, 0.0f);
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
	ImGui::TextDisabled("drag a light to move it");
	if (scene.lights.size() < foto::MAX_LIGHTS && ImGui::Button("add light")) addLight();
	if (!detectedLights.empty()) {
		if (scene.lights.size() < foto::MAX_LIGHTS) ImGui::SameLine();
		if (ImGui::Button("reset lights")) {
			scene.lights = detectedLights;
			scene.lighting.ambient = detectedAmbient;
			selectedLight = -1;
			unsavedChanges = true;
		}
	}

	if (selectedLight < 0 || selectedLight >= static_cast<int>(scene.lights.size())) return;
	recast::SceneLight& light = scene.lights[selectedLight];
	if (scene.lights.size() < foto::MAX_LIGHTS) ImGui::SameLine();
	if (ImGui::Button("remove light")) {
		scene.lights.erase(scene.lights.begin() + selectedLight);
		selectedLight = -1;
		unsavedChanges = true;
		return;
	}

	float strongest = std::max({ light.intensity[0], light.intensity[1], light.intensity[2] });
	float brightness = strongest;
	if (ImGui::SliderFloat("brightness", &brightness, 0.0f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic)) {
		for (float& channel : light.intensity) channel = strongest > 1e-6f ? channel * brightness / strongest : brightness;
		unsavedChanges = true;
	}
}

void Visualizer::updateObjectInteraction() {
	if (!objectMesh) return;

	foto::Window& window = getWindow();
	bool leftDown = window.isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
	Vec3 origin, direction;
	bool haveRay = getCursorRay(origin, direction);
	recast::ObjectTransform& transform = scene.object.transform;

	if (leftDown && !leftMouseWasDown && haveRay) {
		Vec3 worldMin, worldMax;
		recast::objectWorldBounds(transform, objectMesh->getBoundsMin(), objectMesh->getBoundsMax(), worldMin, worldMax);
		float t = 0.0f;
		recast::RayHit hit = supportUnderCursor(origin, direction);
		if (recast::rayAabb(origin, direction, worldMin, worldMax, t) && hit.valid()) {
			draggingObject = true;
			dragOffset = Vec3(transform.position[0] - hit.point.x, 0.0f, transform.position[2] - hit.point.z);
		}
	}
	if (!leftDown) draggingObject = false;

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
		lightingStatus = "export failed: couldn't write " + pngPath;
		return;
	}
	lightingStatus = "exported " + pngPath + " in " + std::to_string(seconds) + " s";
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
		else draw(proxyMeshes[i], matrix, photoMaterial, compositePipeline, foto::ShadowCaster::Static);
	}

	if (objectMesh && !showOriginal) draw(objectMesh, objectTransform(), foto::NO_MATERIAL_OVERRIDE, nullptr, foto::ShadowCaster::Dynamic);

	if (showLights) {
		for (size_t i = 0; i < scene.lights.size(); i++) {
			const recast::SceneLight& light = scene.lights[i];
			if (light.type != "point" && light.type != "directional") continue;
			Vec3 at = markerPosition(light);
			bool selected = static_cast<int>(i) == selectedLight;
			float size = selected ? 0.08f : 0.06f;
			draw(lightMarker, Mat4::translationM(at.x, at.y, at.z) * Mat4::scaleM(size, size, size), selected ? selectedMarkerMaterial : lightMarkerMaterial);
		}
	}

}

void Visualizer::onImGui() {
	ImGui::SetNextWindowPos({ 12.0f, 12.0f }, ImGuiCond_FirstUseEver);
	ImGui::Begin("recast");

	if (!sceneLoaded) {
		ImGui::TextWrapped("no scene loaded: pick a scene photo, or run with a path to scene.json");
		drawPipeline();
		ImGui::End();
		return;
	}

	ImGui::TextDisabled("drag the object, scroll to rotate, shift+scroll to scale");
	ImGui::TextWrapped("%s", lightingStatus.c_str());
	drawPipeline();

	ImGui::BeginDisabled(detectJob.valid());
	if (ImGui::Button("re-detect lighting")) detectLighting();
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button(unsavedChanges ? "save scene *" : "save scene")) saveScene();
	ImGui::SameLine();
	if (ImGui::Button("export image")) exportRequested = true;

	if (objectMesh) {
		recast::ObjectTransform& transform = scene.object.transform;
		if (ImGui::SliderFloat("object size", &transform.scale, 0.1f, 10.0f, "%.2fx", ImGuiSliderFlags_Logarithmic)) {
			transform.scale = std::max(transform.scale, 0.01f);
			recast::resolveCollisions(scene.proxies, transform, objectMesh->getBoundsMin(), objectMesh->getBoundsMax());
			unsavedChanges = true;
		}
	}

	float exposure = getExposure();
	if (ImGui::SliderFloat("exposure", &exposure, -4.0f, 4.0f, "%.1f EV")) setExposure(exposure);

	if (ImGui::Checkbox("edit room", &editRoom) && !editRoom) {
		selectedProxy = -1;
		draggingProxy = false;
	}
	if (editRoom) {
		drawRoomEditor();
		drawRoomOverlay();
	}

	if (openLightEditor) {
		ImGui::SetNextItemOpen(true);
		openLightEditor = false;
	}
	if (ImGui::CollapsingHeader("debug")) {
		if (supportsPathTracing()) {
			ImGui::Checkbox("path tracing", &pathTrace);
			if (isPathTracing()) {
				ImGui::SameLine();
				ImGui::TextDisabled("%u samples", getPathTraceSamples());
				if (ImGui::Checkbox("denoise", &denoise)) setDenoise(denoise);
			}
		}
		else {
			ImGui::TextDisabled("path tracing: this GPU has no ray queries");
		}
		ImGui::Checkbox("original photo", &showOriginal);
		ImGui::Checkbox("show and edit lights", &showLights);
		if (showLights) drawLightEditor();
		ImGui::TextDisabled("%.0f fps, F12 saves a screenshot", ImGui::GetIO().Framerate);
	}

	ImGui::End();
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
	if (pendingObjectReload) loadedObjectPath.clear();
	scenePath = pendingScenePath;
	pendingScenePath.clear();
	loadScene();
	sceneWatcher = foto::FileWatcher(scenePath);
}

void Visualizer::drawPipeline() {
	ImGui::BeginDisabled(pythonJob.running());
	if (ImGui::Button("scene photo...")) {
		std::string photo = pickPhoto("scene photo");
		if (!photo.empty()) {
			std::string name = folderName(photo);
			std::filesystem::path result = std::filesystem::path(RECAST_DATA_DIR) / name / "scene.json";
			std::string dataRoot = nativePath(RECAST_DATA_DIR);
			runPython("building scene " + name, "recast", { "scene", photo, "--name", name, "--data-root", dataRoot, "--overwrite" },
					  nativePath(result), true, false);
			followUpLabel = "finding walls and furniture";
			followUpArguments = { "detect", "--name", name, "--data-root", dataRoot };
		}
	}

	ImGui::BeginDisabled(!sceneLoaded);
	ImGui::SameLine();
	if (ImGui::Button("object photo...")) {
		std::string photo = pickPhoto("object photo");
		if (!photo.empty()) {
			char height[32];
			std::snprintf(height, sizeof(height), "%.3f", std::max(objectHeight, 0.01f));
			std::vector<std::string> arguments = { photo, "--mode", OBJECT_MODES[objectMode], "--height", height,
												   "--out", nativePath(scene.directory) };
			if (objectMode == GLASS_SHAPE) arguments.push_back("--glass");
			runPython("adding object", "recast.object", arguments, scenePath, false, true);
		}
	}
	ImGui::SameLine();
	ImGui::SetNextItemWidth(90.0f);
	ImGui::Combo("##shape", &objectMode, OBJECT_SHAPES, IM_ARRAYSIZE(OBJECT_SHAPES));
	ImGui::SameLine();
	ImGui::SetNextItemWidth(70.0f);
	ImGui::InputFloat("m tall", &objectHeight, 0.0f, 0.0f, "%.2f");
	ImGui::EndDisabled();
	ImGui::EndDisabled();

	if (!pipelineStatus.empty()) ImGui::TextWrapped("%s", pipelineStatus.c_str());
	if (pythonFailed) {
		std::string log = pythonJob.output();
		ImGui::BeginChild("python log", ImVec2(0.0f, 140.0f), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
		ImGui::TextUnformatted(log.c_str(), log.c_str() + log.size());
		ImGui::EndChild();
	}
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
		proxyAlbedoMaterials.push_back(createMaterial({ .baseColor = { albedo[0], albedo[1], albedo[2], 1.0f } }));
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

// Names from detect.json ("chair", "table"...), so the room list and the labels say what each box is
void Visualizer::loadProxyLabels() {
	proxyLabels.clear();
	std::ifstream file(scene.directory / "detect.json");
	if (!file) return;
	try {
		nlohmann::json detect = nlohmann::json::parse(file);
		for (const auto& object : detect.value("objects", nlohmann::json::array())) {
			std::string label = object.value("label", "object");
			const auto& ids = object.value("proxy_ids", nlohmann::json::array());
			for (size_t k = 0; k < ids.size(); k++) {
				proxyLabels[ids[k].get<foto::u32>()] = ids.size() == 1 ? label : label + (k == 0 ? " (seat)" : " (back)");
			}
		}
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
		if (proxy.type != "box") continue;
		float t = 0.0f;
		Vec3 lo(proxy.min[0], proxy.min[1], proxy.min[2]), hi(proxy.max[0], proxy.max[1], proxy.max[2]);
		if (recast::rayAabb(origin, direction, lo, hi, t) && t < bestT) {
			bestT = t;
			best = static_cast<int>(i);
		}
	}
	return best;
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
	if (haveSelection && !draggingProxy) {
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
			for (int a : { 0, 2 }) {
				proxy.min[a] = proxyAtGrab.min[a] + delta[a];
				proxy.max[a] = proxyAtGrab.max[a] + delta[a];
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
		bool selected = static_cast<int>(i) == selectedProxy, hovered = static_cast<int>(i) == hoveredProxy;
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
			if (projectToScreen(Vec3((lo.x + hi.x) * 0.5f, hi.y, (lo.z + hi.z) * 0.5f), at)) {
				std::string name = proxyName(i);
				ImVec2 size = ImGui::CalcTextSize(name.c_str());
				ImVec2 corner0(at.x - size.x * 0.5f - 3.0f, at.y - size.y - 8.0f);
				list->AddRectFilled(corner0, ImVec2(corner0.x + size.x + 6.0f, corner0.y + size.y + 2.0f), IM_COL32(0, 0, 0, 150), 3.0f);
				list->AddText(ImVec2(corner0.x + 3.0f, corner0.y + 1.0f), color, name.c_str());
			}
			if (selected) {
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
	ImGui::TextDisabled("click a piece to select it, drag to move, drag its dots to resize");
	if (ImGui::BeginListBox("##room", ImVec2(-FLT_MIN, 6.5f * ImGui::GetTextLineHeightWithSpacing()))) {
		for (size_t i = 0; i < scene.proxies.size(); i++) {
			const recast::Proxy& proxy = scene.proxies[i];
			if (proxy.type != "box" && !isWall(proxy)) continue;
			std::string label = proxyName(i) + "##" + std::to_string(i);
			if (ImGui::Selectable(label.c_str(), static_cast<int>(i) == selectedProxy)) selectedProxy = static_cast<int>(i);
			if (ImGui::IsItemHovered()) hoveredProxy = static_cast<int>(i);
		}
		ImGui::EndListBox();
	}
	if (ImGui::Button("add box")) {
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
	if (ImGui::Button("delete") && haveSelection) {
		scene.proxies.erase(scene.proxies.begin() + selectedProxy);
		selectedProxy = -1;
		proxyMeshesDirty = true;
		unsavedChanges = true;
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button("reset room")) {
		scene.proxies = detectedProxies;
		selectedProxy = -1;
		proxyMeshesDirty = true;
		unsavedChanges = true;
	}
}
