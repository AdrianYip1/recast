#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "application.hpp"
#include "../platform/window.hpp"
#include "../rhi_vulkan/vkDevice.hpp"

foto::Application::Application(const ApplicationConfig& config) :
	window(std::make_unique<Window>(config.width, config.height, config.title)),
	device(std::make_unique<Device>())
{
	 
}

foto::Application::~Application() {

}

void foto::Application::run() {
	onInit();
	while (!window->shouldClose()) {
		window->pollEvents();
		onUpdate(0.0f);
	}
	onShutdown();
}
