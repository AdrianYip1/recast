#include "visualizer.hpp"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <engine/platform/window.hpp>

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

	if (!scenePath.empty()) {
		loadScene();
		sceneWatcher = foto::FileWatcher(scenePath);
	}
}

void Visualizer::loadScene() {
	try {
		scene = recast::loadSceneFile(scenePath);
		sceneLoaded = true;
		std::cout << "[recast] loaded " << scenePath << ": " << scene.image_size[0] << "x" << scene.image_size[1]
				  << " photo, " << scene.proxies.size() << " proxies, " << scene.lights.size() << " lights" << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "[recast] " << e.what() << std::endl;
	}
}

void Visualizer::onUpdate(foto::f32 deltaTime) {
	if (sceneWatcher.changed()) loadScene();

	cubeAngle += spinSpeed * deltaTime;
	flyCamera(deltaTime);

	bool screenshotKeyDown = getWindow().isKeyDown(GLFW_KEY_F12);
	if (screenshotKeyDown && !screenshotKeyWasDown) screenshot("screenshot.png");
	screenshotKeyWasDown = screenshotKeyDown;

	draw(ground, Mat4::translationM(0.0f, -0.5f, 0.0f), groundMaterial);
	draw(quad, Mat4::translationM(0.9f, 0.1f, -0.6f), checkerMaterial);
	draw(cube, Mat4::translationM(-0.3f, 0.0f, 0.2f) * Mat4::rotateY(cubeAngle));
}

void Visualizer::onImGui() {
	ImGui::SetNextWindowPos({ 12.0f, 12.0f }, ImGuiCond_FirstUseEver);
	ImGui::Begin("foto");
	ImGui::Text("%.1f fps", ImGui::GetIO().Framerate);
	ImGui::SliderFloat("cube spin", &spinSpeed, 0.0f, 3.0f);

	foto::Material groundTint = getMaterial(groundMaterial);
	if (ImGui::ColorEdit3("ground tint", &groundTint.baseColor.x)) {
		updateMaterial(groundMaterial, groundTint);
	}

	if (ImGui::Button("screenshot (F12)")) screenshot("screenshot.png");
	ImGui::TextDisabled("hold right mouse + WASD / QE to fly");
	ImGui::End();
}

void Visualizer::onShutdown() {
	std::cout << "Visualizer application closed" << std::endl;
}
