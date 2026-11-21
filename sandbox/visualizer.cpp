#include "visualizer.hpp"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <engine/platform/window.hpp>
#include <engine/rhi_vulkan/vkPipeline.hpp>

#include "scene/proxyMesh.hpp"
#include "scene/raycast.hpp"
#include "scene/placement.hpp"

#include <chrono>

using enginemath::Vec3;
using enginemath::Mat4;

static const std::vector<foto::Vertex> quadVertices = {
	{{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
	{{0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
	{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
	{{-0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}}
};

static const std::vector<foto::Vertex> groundVertices = {
	{{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 8.0f}},
	{{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {8.0f, 8.0f}},
	{{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {8.0f, 0.0f}},
	{{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}}
};

static const std::vector<foto::u32> quadIndices = { 0, 1, 2, 2, 3, 0 };

Visualizer::Visualizer(const foto::ApplicationConfig& config, std::string scenePath) :
	Application(config), scenePath(std::move(scenePath))
{
}

void Visualizer::onInit() {
	std::cout << "Visualizer application opened" << std::endl;

	foto::u32 checker = loadTexture("textures/checker.png");
	checkerMaterial = createMaterial({ .baseColorTexture = checker });
	groundMaterial = createMaterial({ .baseColor = { 0.55f, 0.55f, 0.6f, 1.0f }, .baseColorTexture = checker });

	quad = createMesh(quadVertices, quadIndices);
	ground = createMesh(groundVertices, quadIndices);
	cube = loadModel("models/cube.glb");

	foto::Camera& camera = getCamera();
	camera.position = { 1.6f, 1.4f, 2.6f };
	camera.lookAt({ 0.0f, 0.3f, 0.0f });

	backgroundPipeline = createPipeline({
		.vertexShaderFilePath = "shaders/fullscreen.vert.spv",
		.fragmentShaderFilePath = "shaders/background.frag.spv",
		.useVertexInput = false,
		.cullMode = VK_CULL_MODE_NONE,
		.depthTest = false,
		.depthWrite = false,
	});
	proxyDepthPipeline = createPipeline({
		.vertexShaderFilePath = "shaders/basic.vert.spv",
		.fragmentShaderFilePath = "shaders/proxy.frag.spv",
		.cullMode = VK_CULL_MODE_NONE,
		.colorWrite = false,
	});
	proxyPipeline = createPipeline({
		.vertexShaderFilePath = "shaders/basic.vert.spv",
		.fragmentShaderFilePath = "shaders/proxy.frag.spv",
		.cullMode = VK_CULL_MODE_NONE,
		.depthWrite = false,
		.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL,
		.alphaBlend = true,
	});
	proxyMaterial = createMaterial({ .baseColor = { 0.1f, 0.5f, 1.0f, proxyOpacity } });

	compositePipeline = createPipeline({
		.vertexShaderFilePath = "shaders/basic.vert.spv",
		.fragmentShaderFilePath = "shaders/composite.frag.spv",
		.cullMode = VK_CULL_MODE_NONE,
	});
	litProxyPipeline = createPipeline({
		.vertexShaderFilePath = "shaders/basic.vert.spv",
		.fragmentShaderFilePath = "shaders/litProxy.frag.spv",
		.cullMode = VK_CULL_MODE_NONE,
	});

	recast::Proxy markerBox;
	markerBox.type = "box";
	markerBox.min = { -0.5f, -0.5f, -0.5f };
	markerBox.max = { 0.5f, 0.5f, 0.5f };
	recast::MeshData markerData = recast::buildProxyMesh(markerBox);
	lightMarker = createMesh(markerData.vertices, markerData.indices);
	lightMarkerMaterial = createMaterial({ .baseColor = { 0.0f, 0.0f, 0.0f, 1.0f }, .emissive = { 4.0f, 3.6f, 1.2f, 0.0f } });

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
			foto::u32 photoTexture = loadTextureHDR(photoPath);
			photoMaterial = createMaterial({ .baseColorTexture = photoTexture });
			loadedPhotoPath = photoPath;
			photoPixels = recast::loadLinearImage(photoPath);
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

	calibrationKey.clear();
	bool hasPointLight = std::any_of(scene.lights.begin(), scene.lights.end(), [](const recast::SceneLight& l) { return l.type == "point"; });
	if (!hasPointLight && !loadedPhotoPath.empty()) startLightingFit(true);

	std::cout << "[recast] loaded " << scenePath << ": " << scene.image_size[0] << "x" << scene.image_size[1]
			  << " photo, " << scene.proxies.size() << " proxies, " << scene.lights.size() << " lights" << std::endl;
}

void Visualizer::applySceneCamera() {
	foto::Camera& camera = getCamera();
	const recast::SceneCamera& source = scene.camera;
	foto::f32 width = static_cast<foto::f32>(scene.image_size[0]);
	foto::f32 height = static_cast<foto::f32>(scene.image_size[1]);

	camera.setIntrinsics({ source.fx, source.fy, source.cx, source.cy, width, height });

	Mat4 orientation = Mat4::identity();
	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			orientation.m[col][row] = source.rotation[row][col];
		}
	}
	camera.setOrientation(orientation);
	camera.position = Vec3(source.position[0], source.position[1], source.position[2]);

	setViewportAspect(width / height);
}

void Visualizer::onUpdate(foto::f32 deltaTime) {
	if (sceneWatcher.changed()) loadScene();

	bool screenshotKeyDown = getWindow().isKeyDown(GLFW_KEY_F12);
	if (screenshotKeyDown && !screenshotKeyWasDown) screenshot("screenshot.png");
	screenshotKeyWasDown = screenshotKeyDown;

	if (sceneLoaded) {
		if (freeCamera) flyCamera(deltaTime);
		else applySceneCamera();
		finishLightingFit();
		applySceneLighting();
		updateLightPlacement();
		if (placingLight < 0) updateObjectInteraction();
		updateCalibration(deltaTime);
		drawPhotoView();
	}
	else {
		flyCamera(deltaTime);
		cubeAngle += spinSpeed * deltaTime;
		drawDemo();
	}
}

void Visualizer::drawPhotoView() {
	if (!loadedPhotoPath.empty()) drawFullscreen(backgroundPipeline, photoMaterial);

	for (size_t i = 0; i < proxyMeshes.size(); i++) {
		switch (viewMode) {
		case ViewMode::Composite:
			draw(proxyMeshes[i], Mat4::identity(), photoMaterial, compositePipeline, foto::ShadowCaster::Static);
			break;
		case ViewMode::Photo:
			draw(proxyMeshes[i], Mat4::identity(), proxyMaterial, proxyDepthPipeline, foto::ShadowCaster::Static);
			break;
		case ViewMode::Lighting:
			draw(proxyMeshes[i], Mat4::identity(), proxyAlbedoMaterials[i], litProxyPipeline, foto::ShadowCaster::Static);
			break;
		}
	}

	if (objectMesh && viewMode != ViewMode::Photo) {
		draw(objectMesh, objectTransform(), foto::NO_MATERIAL_OVERRIDE, nullptr, foto::ShadowCaster::Dynamic,
			 autoCalibrate ? calibration : Vec3(1.0f, 1.0f, 1.0f));
	}

	if (showLights) {
		for (const recast::SceneLight& light : scene.lights) {
			if (light.type != "point") continue;
			draw(lightMarker, Mat4::translationM(light.position[0], light.position[1], light.position[2]) * Mat4::scaleM(0.06f, 0.06f, 0.06f),
				 lightMarkerMaterial);
		}
	}

	if (showProxies) {
		for (foto::Mesh* mesh : proxyMeshes) draw(mesh, Mat4::identity(), proxyMaterial, proxyPipeline);
	}
}

void Visualizer::applySceneLighting() {
	std::vector<foto::PointLight> lights;
	for (const recast::SceneLight& light : scene.lights) {
		if (light.type != "point") continue;
		lights.push_back({
			.position = Vec3(light.position[0], light.position[1], light.position[2]),
			.intensity = Vec3(light.intensity[0], light.intensity[1], light.intensity[2]),
		});
	}
	setLights(lights);

	const recast::Vec3f& ambient = scene.lighting.ambient;
	setAmbient(Vec3(ambient[0], ambient[1], ambient[2]));
}

void Visualizer::updateCalibration(foto::f32 deltaTime) {
	if (!objectMesh || photoPixels.rgb.empty() || !autoCalibrate) return;

	const recast::ObjectTransform& transform = scene.object.transform;
	std::vector<float> key = { transform.position[0], transform.position[1], transform.position[2], transform.scale,
							   scene.lighting.ambient[0], scene.lighting.ambient[1], scene.lighting.ambient[2] };
	for (const recast::SceneLight& light : scene.lights) {
		key.insert(key.end(), light.position.begin(), light.position.end());
		key.insert(key.end(), light.intensity.begin(), light.intensity.end());
	}

	if (key != calibrationKey) {
		calibrationKey = key;
		Vec3 worldMin, worldMax;
		recast::objectWorldBounds(transform, objectMesh->getBoundsMin(), objectMesh->getBoundsMax(), worldMin, worldMax);
		float footprint = std::max(worldMax.x - worldMin.x, worldMax.z - worldMin.z);
		Vec3 base((worldMin.x + worldMax.x) * 0.5f, transform.position[1], (worldMin.z + worldMax.z) * 0.5f);
		calibrationTarget = recast::localLightingCalibration(scene, photoPixels, base, std::max(footprint * 1.5f, 0.4f));
	}

	float blend = std::min(1.0f, deltaTime * 8.0f);
	calibration = calibration + (calibrationTarget - calibration) * blend;
}

void Visualizer::updateLightPlacement() {
	if (placingLight < 0) return;

	bool leftDown = getWindow().isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
	Vec3 origin, direction;
	if (leftDown && !leftMouseWasDown && getCursorRay(origin, direction)) {
		recast::RayHit hit = recast::raycastProxies(scene.proxies, origin, direction);
		if (hit.valid() && placingLight < static_cast<int>(scene.lights.size())) {
			Vec3 position = hit.point + hit.normal * 0.3f;
			scene.lights[placingLight].position = { position.x, position.y, position.z };
			unsavedChanges = true;
		}
		placingLight = -1;
	}
	leftMouseWasDown = leftDown;
}

void Visualizer::startLightingFit(bool refinePositions) {
	if (fitJob.valid() || loadedPhotoPath.empty()) return;

	std::vector<Vec3> positions;
	for (const recast::SceneLight& light : scene.lights) {
		if (light.type == "point") positions.push_back(Vec3(light.position[0], light.position[1], light.position[2]));
	}
	if (!refinePositions && positions.empty()) {
		fitStatus = "add a light first, or use fit lighting";
		return;
	}

	fitStatus = "fitting...";
	recast::SceneFile sceneCopy = scene;
	std::string photoPath = loadedPhotoPath;
	fitJob = std::async(std::launch::async, [sceneCopy, photoPath, positions, refinePositions]() mutable {
		FitJob job;
		try {
			recast::LinearImage photo = recast::loadLinearImage(photoPath);
			std::vector<recast::LightingSample> samples = recast::collectLightingSamples(sceneCopy, photo, 8);
			job.positions = positions;
			job.fit = refinePositions ? recast::fitLighting(sceneCopy, samples, job.positions)
									  : recast::fitLightIntensities(sceneCopy, samples, job.positions);
		}
		catch (const std::exception& e) {
			job.error = e.what();
		}
		return job;
	});
}

void Visualizer::finishLightingFit() {
	if (!fitJob.valid() || fitJob.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;

	FitJob job = fitJob.get();
	if (!job.error.empty()) {
		fitStatus = "fit failed: " + job.error;
		return;
	}

	std::vector<recast::SceneLight> lights;
	for (const recast::SceneLight& light : scene.lights) {
		if (light.type != "point") lights.push_back(light);
	}
	for (size_t i = 0; i < job.positions.size(); i++) {
		recast::SceneLight light;
		light.position = { job.positions[i].x, job.positions[i].y, job.positions[i].z };
		light.intensity = { job.fit.intensities[i].x, job.fit.intensities[i].y, job.fit.intensities[i].z };
		lights.push_back(light);
	}
	scene.lights = lights;
	scene.lighting.ambient = { job.fit.ambient.x, job.fit.ambient.y, job.fit.ambient.z };
	unsavedChanges = true;

	char buffer[128];
	std::snprintf(buffer, sizeof(buffer), "fitted %zu light(s), ambient (%.3f, %.3f, %.3f)",
				  job.positions.size(), job.fit.ambient.x, job.fit.ambient.y, job.fit.ambient.z);
	fitStatus = buffer;
}

void Visualizer::drawLightingPanel() {
	ImGui::SeparatorText("lighting");

	int mode = static_cast<int>(viewMode);
	ImGui::RadioButton("composite", &mode, 0); ImGui::SameLine();
	ImGui::RadioButton("photo", &mode, 1); ImGui::SameLine();
	ImGui::RadioButton("estimated lighting", &mode, 2);
	viewMode = static_cast<ViewMode>(mode);
	ImGui::Checkbox("show lights", &showLights);
	ImGui::Checkbox("auto calibrate object lighting", &autoCalibrate);
	if (autoCalibrate) ImGui::TextDisabled("local calibration (%.2f, %.2f, %.2f)", calibration.x, calibration.y, calibration.z);

	unsavedChanges |= ImGui::DragFloat3("ambient", scene.lighting.ambient.data(), 0.005f, 0.0f, 10.0f);

	for (size_t i = 0; i < scene.lights.size(); i++) {
		recast::SceneLight& light = scene.lights[i];
		if (light.type != "point") continue;

		ImGui::PushID(static_cast<int>(i));
		ImGui::Text("light %zu", i);
		unsavedChanges |= ImGui::DragFloat3("position", light.position.data(), 0.01f);
		unsavedChanges |= ImGui::DragFloat3("intensity", light.intensity.data(), 0.01f, 0.0f, 1000.0f);
		if (ImGui::Button(placingLight == static_cast<int>(i) ? "click the photo..." : "place by clicking")) placingLight = static_cast<int>(i);
		ImGui::SameLine();
		if (ImGui::Button("remove")) {
			scene.lights.erase(scene.lights.begin() + i);
			unsavedChanges = true;
			placingLight = -1;
			ImGui::PopID();
			break;
		}
		ImGui::PopID();
	}

	if (scene.lights.size() < foto::MAX_LIGHTS && ImGui::Button("add light")) {
		recast::SceneLight light;
		light.position = { scene.camera.position[0], scene.camera.position[1] + 0.5f, scene.camera.position[2] - 1.0f };
		light.intensity = { 2.0f, 2.0f, 2.0f };
		scene.lights.push_back(light);
		placingLight = static_cast<int>(scene.lights.size() - 1);
		unsavedChanges = true;
	}

	bool fitting = fitJob.valid();
	ImGui::BeginDisabled(fitting);
	if (ImGui::Button("fit lighting to photo")) startLightingFit(true);
	ImGui::SameLine();
	if (ImGui::Button("fit intensities only")) startLightingFit(false);
	ImGui::EndDisabled();
	if (!fitStatus.empty()) ImGui::TextWrapped("%s", fitStatus.c_str());
}

Mat4 Visualizer::objectTransform() const {
	return recast::objectMatrix(scene.object.transform);
}

bool Visualizer::cursorHitsObject(const Vec3& origin, const Vec3& direction) {
	Vec3 worldMin, worldMax;
	recast::objectWorldBounds(scene.object.transform, objectMesh->getBoundsMin(), objectMesh->getBoundsMax(), worldMin, worldMax);
	float t = 0.0f;
	return recast::rayAabb(origin, direction, worldMin, worldMax, t);
}

void Visualizer::resolveCollisions() {
	recast::resolveCollisions(scene.proxies, scene.object.transform, objectMesh->getBoundsMin(), objectMesh->getBoundsMax());
}

void Visualizer::updateObjectInteraction() {
	if (!objectMesh) return;

	foto::Window& window = getWindow();
	bool leftDown = window.isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT);
	Vec3 origin, direction;
	bool haveRay = getCursorRay(origin, direction);
	recast::ObjectTransform& transform = scene.object.transform;
	constexpr float supportNormalY = 0.7f;

	if (leftDown && !leftMouseWasDown && haveRay && cursorHitsObject(origin, direction)) {
		recast::RayHit hit = recast::raycastProxies(scene.proxies, origin, direction, supportNormalY);
		if (hit.valid()) {
			draggingObject = true;
			dragOffset = Vec3(transform.position[0] - hit.point.x, 0.0f, transform.position[2] - hit.point.z);
		}
	}
	if (!leftDown) draggingObject = false;

	if (draggingObject && haveRay) {
		recast::RayHit hit = recast::raycastProxies(scene.proxies, origin, direction, supportNormalY);
		if (hit.valid()) {
			transform.position = { hit.point.x + dragOffset.x, hit.point.y, hit.point.z + dragOffset.z };
			resolveCollisions();
			unsavedChanges = true;
		}
	}

	float scroll = window.getScrollDelta();
	if (scroll != 0.0f && haveRay) {
		if (window.isKeyDown(GLFW_KEY_LEFT_SHIFT)) transform.scale *= std::pow(1.1f, scroll);
		else transform.rotation_y_deg = std::fmod(transform.rotation_y_deg + scroll * 15.0f + 360.0f, 360.0f);
		resolveCollisions();
		unsavedChanges = true;
	}

	leftMouseWasDown = leftDown;
}

void Visualizer::saveScene() {
	try {
		recast::saveSceneFile(scene, scenePath);
		sceneWatcher = foto::FileWatcher(scenePath);
		unsavedChanges = false;
		std::cout << "[recast] saved " << scenePath << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "[recast] " << e.what() << std::endl;
	}
}

void Visualizer::drawDemo() {
	draw(ground, Mat4::translationM(0.0f, -0.5f, 0.0f), groundMaterial);
	draw(quad, Mat4::translationM(0.9f, 0.1f, -0.6f), checkerMaterial);
	draw(cube, Mat4::translationM(-0.3f, 0.0f, 0.2f) * Mat4::rotateY(cubeAngle));
}

void Visualizer::onImGui() {
	ImGui::SetNextWindowPos({ 12.0f, 12.0f }, ImGuiCond_FirstUseEver);
	ImGui::Begin("foto");
	ImGui::Text("%.1f fps", ImGui::GetIO().Framerate);

	if (sceneLoaded) {
		ImGui::Text("%s", scenePath.c_str());
		ImGui::Checkbox("show proxies", &showProxies);
		if (ImGui::SliderFloat("proxy opacity", &proxyOpacity, 0.0f, 1.0f)) {
			foto::Material material = getMaterial(proxyMaterial);
			material.baseColor.w = proxyOpacity;
			updateMaterial(proxyMaterial, material);
		}
		ImGui::Checkbox("free camera", &freeCamera);

		if (objectMesh) {
			ImGui::SeparatorText("object");
			recast::ObjectTransform& transform = scene.object.transform;
			unsavedChanges |= ImGui::DragFloat3("position", transform.position.data(), 0.01f);
			unsavedChanges |= ImGui::SliderFloat("rotation", &transform.rotation_y_deg, 0.0f, 360.0f, "%.0f deg");
			unsavedChanges |= ImGui::DragFloat("scale", &transform.scale, 0.01f, 0.01f, 100.0f);
			ImGui::TextDisabled("drag with left mouse, scroll to rotate, shift+scroll to scale");
		}
		drawLightingPanel();
		if (ImGui::Button(unsavedChanges ? "save scene *" : "save scene")) saveScene();
	}
	else {
		ImGui::SliderFloat("cube spin", &spinSpeed, 0.0f, 3.0f);
		foto::Material groundTint = getMaterial(groundMaterial);
		if (ImGui::ColorEdit3("ground tint", &groundTint.baseColor.x)) {
			updateMaterial(groundMaterial, groundTint);
		}
	}

	float exposure = getExposure();
	if (ImGui::SliderFloat("exposure (EV)", &exposure, -4.0f, 4.0f)) setExposure(exposure);
	Vec3 balance = getWhiteBalance();
	if (ImGui::ColorEdit3("white balance", &balance.x)) setWhiteBalance(balance);

	if (ImGui::Button("screenshot (F12)")) screenshot("screenshot.png");
	if (!sceneLoaded || freeCamera) ImGui::TextDisabled("hold right mouse + WASD / QE to fly");
	ImGui::End();
}

void Visualizer::onShutdown() {
	std::cout << "Visualizer application closed" << std::endl;
}
