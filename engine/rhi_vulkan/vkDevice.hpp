// instances, devices, queues
#pragma once

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

namespace foto {

	struct QueueInfos {
		std::optional<u32> graphicsQueueIdx;
		std::optional<u32> computeQueueIdx;
	};

	class Device {
	public :
		Device();
		~Device();
		Device(const Device& other) = delete;
		Device& operator=(const Device& other) = delete;
	private:
		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		VkDevice logicalDevice = VK_NULL_HANDLE;

		static VkInstance createInstance();
		static VkDebugUtilsMessengerEXT createDebugMessenger(VkInstance instance);
		
		static VkPhysicalDevice selectPhysicalDevice(VkInstance instance);
		static std::optional<foto::QueueInfos> findQueueFamilies(VkPhysicalDevice device);
		static bool extensionsSuitable(VkPhysicalDevice device);
		static VkDevice selectLogicalDevice(VkPhysicalDevice physicalDevice);
	};
} // namespace foto