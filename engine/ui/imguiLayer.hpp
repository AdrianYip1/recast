#pragma once

#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>

namespace foto {
	class Window;
	class Device;
	class Swapchain;

	class ImGuiLayer {
	public:
		ImGuiLayer(Window& window, Device& device, Swapchain& swapchain, VkFormat depthFormat);
		~ImGuiLayer();
		ImGuiLayer(const ImGuiLayer& other) = delete;
		ImGuiLayer& operator=(const ImGuiLayer& other) = delete;

		void beginFrame();
		void endFrame();
		void record(VkCommandBuffer commandBuffer);

		bool wantsMouse();
		bool wantsKeyboard();
	};
} // namespace foto
