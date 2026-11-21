#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"
#include "vkSwapchain.hpp"

namespace foto {
	class Render {
	public:
		Render(Device& device, Swapchain& swapchain);
		~Render();
		Render(const Render& other) = delete;
		Render& operator=(const Render& other) = delete;

	private:

	};
} // namespace foto