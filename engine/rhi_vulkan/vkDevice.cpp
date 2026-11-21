#include "vkDevice.hpp"

#ifdef NDEBUG
static constexpr bool enableValidation = false;
#else
static constexpr bool enableValidation = true;
#endif

// Extensions for the physical device
static const std::vector<const char*> requiredDeviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

static VkDebugUtilsMessengerCreateInfoEXT debugMessengerInfo() {
	return {
		.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
		.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
			| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
		.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT,
			VkDebugUtilsMessageTypeFlagsEXT,
			const VkDebugUtilsMessengerCallbackDataEXT* data,
			void*) -> VkBool32 {
			std::cerr << "[vulkan] " << data->pMessage << '\n';
			return VK_FALSE;
		},
	};
}

foto::Device::Device() :
	instance(createInstance()),
	debugMessenger(createDebugMessenger(instance)),
	physicalDevice(selectPhysicalDevice(instance)),
	logicalDevice(selectLogicalDevice(physicalDevice))
{

}

foto::Device::~Device() {
	if (debugMessenger) {
		auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
			vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
		destroyMessenger(instance, debugMessenger, nullptr);
	}
	vkDestroyInstance(instance, nullptr);
}

VkInstance foto::Device::createInstance() {
	VkInstance instance = VK_NULL_HANDLE;

	u32 glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	if (!glfwExtensions) throw std::runtime_error("ERROR: vkDevice.cpp -> createInstance()");
	
	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
	std::vector<const char*> layers;

	if (enableValidation) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		layers.push_back("VK_LAYER_KHRONOS_validation");

		u32 availableCount = 0;
		vkEnumerateInstanceLayerProperties(&availableCount, nullptr);
		std::vector<VkLayerProperties> available(availableCount);
		vkEnumerateInstanceLayerProperties(&availableCount, available.data());

		for (const char* layer : layers) {
			bool found = std::any_of(available.begin(), available.end(),
				[&](const VkLayerProperties& p) { return std::strcmp(p.layerName, layer) == 0; });
			if (!found) {
				throw std::runtime_error(std::string("ERROR: vkDevice.cpp -> createInstance()") + layer);
			}
		}
	}

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = debugMessengerInfo();

	VkApplicationInfo appInfo = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "Photo Visualizer Software",
		.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		.pEngineName = "Vulkan Custom Engine",
		.engineVersion = VK_MAKE_VERSION(1, 0, 0),
		.apiVersion = VK_API_VERSION_1_3
	};
	VkInstanceCreateInfo instanceCreateInfo{
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pNext = enableValidation ? &debugCreateInfo : nullptr,
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = static_cast<u32>(layers.size()),
		.ppEnabledLayerNames = layers.data(),
		.enabledExtensionCount = static_cast<u32>(extensions.size()),
		.ppEnabledExtensionNames = extensions.data(),
	};

	if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> createInstance()");
	}

	return instance;
}

VkDebugUtilsMessengerEXT foto::Device::createDebugMessenger(VkInstance instance) {
	if (!enableValidation) return VK_NULL_HANDLE;
	

	auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
		vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));

	VkDebugUtilsMessengerCreateInfoEXT createInfo = debugMessengerInfo();
	VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
	if (!createMessenger || createMessenger(instance, &createInfo, nullptr, &messenger) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> createDebugMessenger()");
	}
	return messenger;
}

VkPhysicalDevice foto::Device::selectPhysicalDevice(VkInstance instance) {
	u32 deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
	if (deviceCount == 0) throw std::runtime_error("ERROR: vkDevice.cpp -> selectPhysicalDevice(): no Vulkan GPU found");
	
	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	VkPhysicalDevice chosenDevice = VK_NULL_HANDLE;
	for (auto device : devices) {
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(device, &properties);
		if (properties.apiVersion < VK_API_VERSION_1_3) continue;

		// Check for Vulkan Features
		VkPhysicalDeviceVulkan13Features features13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
		VkPhysicalDeviceFeatures2 features2 = { 
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, 
			.pNext = &features13
		};
		vkGetPhysicalDeviceFeatures2(device, &features2);
		if (!features13.dynamicRendering || !features13.synchronization2) continue;

		// Check for queues
		if (!findQueueFamilies(device).has_value()) continue;

		// Check for extensions
		if (!extensionsSuitable(device)) continue;

		chosenDevice = device;
		// Use discrete if found right way
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
	} 

	if (chosenDevice == VK_NULL_HANDLE) throw std::runtime_error("ERROR: vkDevice.cpp -> selectPhysicalDevice()");
	return chosenDevice;
}

std::optional<foto::QueueInfos> foto::Device::findQueueFamilies(VkPhysicalDevice device) {
	u32 familyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
	std::vector<VkQueueFamilyProperties> properties(familyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, properties.data());

	std::optional<u32> graphics, compute;
	for (u32 i = 0; i < familyCount; i++) {
		if (!graphics && properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) graphics = i;
		if (!compute && properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) compute = i;
	}

	if (!graphics || !compute) return std::nullopt;
	return QueueInfos{
		.graphicsQueueIdx = graphics,
		.computeQueueIdx = compute
	};
}

bool foto::Device::extensionsSuitable(VkPhysicalDevice device) {
	u32 deviceExtensionCount = 0;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &deviceExtensionCount, nullptr);

	std::vector<VkExtensionProperties> extensions(deviceExtensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &deviceExtensionCount, extensions.data());

	return std::all_of(requiredDeviceExtensions.begin(), requiredDeviceExtensions.end(),
		[&](const char* required) {
			return std::any_of(extensions.begin(), extensions.end(),
				[&](const VkExtensionProperties& e) {
					return std::strcmp(e.extensionName, required) == 0;
				});
		});
}

VkDevice foto::Device::selectLogicalDevice(VkPhysicalDevice physicalDevice) {
	VkDevice logicalDevice = VK_NULL_HANDLE;

	QueueInfos queues = *findQueueFamilies(physicalDevice);



	VkDeviceCreateInfo createInfo = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		

	};

	if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> selectLogicalDevice()");
	}

	return logicalDevice;
}