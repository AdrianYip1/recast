#include "visualizer.hpp"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <engine/platform/window.hpp>
#include <engine/rhi_vulkan/vkPipeline.hpp>

#include "scene/proxyMesh.hpp"
#include "scene/raycast.hpp"
#include "scene/placement.hpp"

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
		}
		catch (const std::exception& e) {
			std::cerr << "[recast] " << e.what() << std::endl;
		}
	}

	for (foto::Mesh* mesh : proxyMeshes) destroyMesh(mesh);
	proxyMeshes.clear();
	for (const recast::Proxy& proxy : loaded.proxies) {
		recast::MeshData data = recast::buildProxyMesh(proxy);
		if (!data.indices.empty()) proxyMeshes.push_back(createMesh(data.vertices, data.indices));
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
		updateObjectInteraction();
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

	for (foto::Mesh* mesh : proxyMeshes) draw(mesh, Mat4::identity(), proxyMaterial, proxyDepthPipeline);
	if (objectMesh) draw(objectMesh, objectTransform());
	if (showProxies) {
		for (foto::Mesh* mesh : proxyMeshes) draw(mesh, Mat4::identity(), proxyMaterial, proxyPipeline);
	}
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
