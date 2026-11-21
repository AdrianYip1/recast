// instances, devices, queues
#pragma once

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

namespace foto {
	class Device {
	public :
		Device();
		~Device();
		Device(const Device& other) = delete;
		Device& operator=(const Device& other) = delete;
	private:
		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

		void createInstance();
	};
} // namespace foto