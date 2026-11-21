#include "vkDevice.hpp"

#ifdef NDEBUG
static constexpr bool enableValidation = false;
#else
static constexpr bool enableValidation = true;
#endif

foto::Device::Device() {
	createInstance();
}

foto::Device::~Device() {
	if (debugMessenger) {
		auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
			vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
		destroyMessenger(instance, debugMessenger, nullptr);
	}
	vkDestroyInstance(instance, nullptr);
}

void foto::Device::createInstance() {
	u32 glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	if (!glfwExtensions) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> createInstance()");
	}
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

	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{
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

	if (enableValidation) {
		auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
			vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
		if (!createMessenger || createMessenger(instance, &debugCreateInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("ERROR: vkDevice.cpp -> createInstance(): failed to create debug messenger");
		}
	}
}
