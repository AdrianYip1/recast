#pragma once

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

#include "vkDevice.hpp"
#include "vkSwapchain.hpp"

namespace foto {
	class Pipeline {
	public:
		Pipeline(Device& device, Swapchain& swapchain);
		~Pipeline();
		Pipeline(const Pipeline& other) = delete;
		Pipeline& operator=(const Pipeline& other) = delete;

	private:
		static std::vector<VkShaderModule> createShaderModules(Device& device, const std::vector<std::vector<char>>& code);
	};
} // namespace foto