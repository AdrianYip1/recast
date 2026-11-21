#pragma once

#include <engine/app/application.hpp>

class Visualizer : public foto::Application {
public:
	Visualizer(const foto::ApplicationConfig& config);

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
};
