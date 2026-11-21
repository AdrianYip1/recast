// instances, devices, queues
#pragma once

#include <vulkan/vulkan.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

struct GLFWwindow;

namespace foto {

	struct QueueInfos {
		std::optional<u32> graphicsQueueIdx;
		std::optional<u32> presentQueueIdx;
		std::optional<u32> computeQueueIdx;
	};

	struct SwapchainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	// Device functions of VK_KHR_acceleration_structure (not exported by the loader)
	struct RayTracingFunctions {
		PFN_vkCreateAccelerationStructureKHR createAccelerationStructure = nullptr;
		PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure = nullptr;
		PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
		PFN_vkCmdBuildAccelerationStructuresKHR cmdBuild = nullptr;
		PFN_vkGetAccelerationStructureDeviceAddressKHR getAddress = nullptr;
	};

	class Device {
	public :
		Device(GLFWwindow* window);
		~Device();
		Device(const Device& other) = delete;
		Device& operator=(const Device& other) = delete;

		VkInstance getInstance() { return instance; };
		VkSurfaceKHR getSurface() { return surface; };
		VkPhysicalDevice getPhysicalDevice() { return physicalDevice; };
		VkDevice getDevice() { return logicalDevice; };

		u32 getGraphicsQueueFamily() { return queueInfos.graphicsQueueIdx.value(); };
		VkQueue getGraphicsQ() { return graphicsQueue; };
		VkQueue getPresentQ() { return presentQueue; };
		VkQueue getComputeQ() { return computeQueue; };

		// Index of a memory type allowed by typeFilter that has all the requested properties
		u32 findMemoryType(u32 typeFilter, VkMemoryPropertyFlags properties);

		// Records commands into a one-time command buffer, submits to the graphics queue and waits.
		// For uploads/transitions at load time, not per frame.
		void immediateSubmit(VkCommandPool commandPool, const std::function<void(VkCommandBuffer)>& record);

		// GPU's anisotropy limit, or 1.0 if samplerAnisotropy isn't supported (it's enabled whenever it is)
		f32 getMaxSamplerAnisotropy();

		// Whether LINE/POINT polygon modes work (fillModeNonSolid is enabled whenever it's supported)
		bool supportsWireframe();

		// Acceleration structures + ray queries, enabled whenever the GPU has them
		bool supportsRayTracing() { return rayTracing; };
		const RayTracingFunctions& getRayTracingFunctions() { return rayTracingFunctions; };
		VkDeviceSize getScratchAlignment() { return scratchAlignment; };

		static SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);
		static std::optional<foto::QueueInfos> findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);

	private:
		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
		VkSurfaceKHR surface = VK_NULL_HANDLE;
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		QueueInfos queueInfos;
		bool rayTracing = false;
		VkDevice logicalDevice = VK_NULL_HANDLE;
		RayTracingFunctions rayTracingFunctions;
		VkDeviceSize scratchAlignment = 256;

		VkQueue graphicsQueue = VK_NULL_HANDLE;
		VkQueue presentQueue = VK_NULL_HANDLE;
		VkQueue computeQueue = VK_NULL_HANDLE;

		static VkInstance createInstance();
		static VkDebugUtilsMessengerEXT createDebugMessenger(VkInstance instance);
		
		static VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow* window);
		static VkPhysicalDevice selectPhysicalDevice(VkInstance instance, VkSurfaceKHR surface);
		static bool extensionsSuitable(VkPhysicalDevice device, std::span<const char* const> required);
		static bool rayTracingSuitable(VkPhysicalDevice device);
		static VkDevice selectLogicalDevice(VkPhysicalDevice physicalDevice, const QueueInfos& queueInfo, bool rayTracing);
	};
} // namespace foto