#include "visualizer.hpp"

Visualizer::Visualizer(const foto::ApplicationConfig& config) :
	Application(config)
{
}

void Visualizer::onInit() {
	std::cout << "Visualizer application opened" << std::endl;
}

void Visualizer::onUpdate(foto::f32 deltaTime) {

}

void Visualizer::onShutdown() {
	std::cout << "Visualizer application closed" << std::endl;
}
