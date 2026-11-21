#include "visualizer.hpp"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <engine/platform/window.hpp>
#include <engine/rhi_vulkan/vkPipeline.hpp>

#include "scene/proxyMesh.hpp"
#include "scene/raycast.hpp"
#include "scene/placement.hpp"

#include <cstdlib>
#include <ctime>

using enginemath::Vec3;
using enginemath::Mat4;

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
	lightMarkerMaterial = createMaterial({ .baseColor = { 0.0f, 0.0f, 0.0f, 1.0f }, .emissive = { 4.0f, 3.6f, 1.2f, 0.0f } });
	selectedMarkerMaterial = createMaterial({ .baseColor = { 0.0f, 0.0f, 0.0f, 1.0f }, .emissive = { 0.8f, 3.0f, 4.0f, 0.0f } });

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

	for (foto::Mesh* mesh : proxyMeshes) destroyMesh(mesh);
	proxyMeshes.clear();
	proxyAlbedoMaterials.clear();
	for (const recast::Proxy& proxy : loaded.proxies) {
		recast::MeshData data = recast::buildProxyMesh(proxy);
		if (data.indices.empty()) continue;
		proxyMeshes.push_back(createMesh(data.vertices, data.indices));
		const recast::Vec3f& albedo = proxy.material.albedo;
		proxyAlbedoMaterials.push_back(createMaterial({ .baseColor = { albedo[0], albedo[1], albedo[2], 1.0f } }));
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
	unsavedChanges = false;
	draggingObject = false;
	applySceneCamera();

	if (!photoPixels.rgb.empty()) {
		std::vector<recast::LightingSample> coarse = recast::collectLightingSamples(scene, photoPixels, 16);
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
	recast::SceneFile sceneCopy = scene;
	recast::LinearImage photo = photoPixels;
	detectJob = std::async(std::launch::async, [sceneCopy, photo]() {
		DetectJob job;
		try {
			std::vector<recast::LightingSample> samples = recast::collectLightingSamples(sceneCopy, photo, 8);
			job.fit = recast::autoFitLighting(sceneCopy, photo, samples, job.lights);
		}
		catch (const std::exception& e) {
			job.error = e.what();
		}
		return job;
	});
}

void Visualizer::finishLightingDetection() {
	if (!detectJob.valid() || detectJob.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;

	DetectJob job = detectJob.get();
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
	unsavedChanges = true;

	int seconds = static_cast<int>(std::chrono::duration<double>(std::chrono::steady_clock::now() - detectStarted).count());
	lightingStatus = "detected " + std::to_string(job.lights.size() - directionalCount) + " point + " +
					 std::to_string(directionalCount) + " directional light(s) in " + std::to_string(seconds) + " s";
	std::cout << "[recast] " << lightingStatus << std::endl;
}

void Visualizer::onUpdate(foto::f32 deltaTime) {
	if (sceneWatcher.changed()) loadScene();

	bool screenshotKeyDown = getWindow().isKeyDown(GLFW_KEY_F12);
	if (screenshotKeyDown && !screenshotKeyWasDown) screenshot("screenshot.png");
	screenshotKeyWasDown = screenshotKeyDown;

	if (!sceneLoaded) return;

	applySceneCamera();
	finishLightingDetection();
	applySceneLighting();
	if (!(showLights && updateLightEditing())) updateObjectInteraction();
	setPathTracing(pathTrace && !showOriginal && !showProxies && !showLights && supportsPathTracing(), photoMaterial);
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
		if (showOriginal) draw(proxyMeshes[i], Mat4::identity(), overlayMaterial, proxyDepthPipeline, foto::ShadowCaster::Static);
		else if (isPathTracing()) draw(proxyMeshes[i], Mat4::identity(), proxyAlbedoMaterials[i], nullptr, foto::ShadowCaster::Static);
		else draw(proxyMeshes[i], Mat4::identity(), photoMaterial, compositePipeline, foto::ShadowCaster::Static);
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

	if (showProxies) {
		for (foto::Mesh* mesh : proxyMeshes) draw(mesh, Mat4::identity(), overlayMaterial, proxyOverlayPipeline);
	}
}

void Visualizer::onImGui() {
	ImGui::SetNextWindowPos({ 12.0f, 12.0f }, ImGuiCond_FirstUseEver);
	ImGui::Begin("recast");

	if (!sceneLoaded) {
		ImGui::TextWrapped("no scene loaded: run with a path to scene.json");
		ImGui::End();
		return;
	}

	ImGui::TextDisabled("drag the object, scroll to rotate, shift+scroll to scale");
	ImGui::TextWrapped("%s", lightingStatus.c_str());

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
		ImGui::Checkbox("show room proxies", &showProxies);
		ImGui::Checkbox("show and edit lights", &showLights);
		if (showLights) ImGui::TextDisabled("drag a light to move it");
		ImGui::TextDisabled("%.0f fps, F12 saves a screenshot", ImGui::GetIO().Framerate);
	}

	ImGui::End();
}
