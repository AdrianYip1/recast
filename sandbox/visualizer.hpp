#pragma once

#include <engine/app/application.hpp>

class Visualizer : public foto::Application {
public:
	Visualizer(const foto::ApplicationConfig& config);

protected:
	void onInit() override;
	void onUpdate(foto::f32 deltaTime) override;
	void onShutdown() override;
};
