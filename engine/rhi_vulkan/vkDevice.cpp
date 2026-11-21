#include "vkDevice.hpp"
#include "vkRenderTypes.hpp"

#ifdef FOTO_DEBUG
static constexpr bool enableValidation = true;
#else
static constexpr bool enableValidation = false;
#endif


#include <GLFW/glfw3.h>

// Extensions for the physical device
static const std::vector<const char*> requiredDeviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

static const std::vector<const char*> rayTracingExtensions = {
	VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	VK_KHR_RAY_QUERY_EXTENSION_NAME,
	VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
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

foto::Device::Device(GLFWwindow* window) :
	instance(createInstance()),
	debugMessenger(createDebugMessenger(instance)),
	surface(createSurface(instance, window)),
	physicalDevice(selectPhysicalDevice(instance, surface)),
	queueInfos(findQueueFamilies(physicalDevice, surface).value()),
	rayTracing(rayTracingSuitable(physicalDevice)),
	textureBinding(chooseTextureBinding(physicalDevice)),
	logicalDevice(selectLogicalDevice(physicalDevice, queueInfos, rayTracing, textureBinding))
{
	vkGetDeviceQueue(logicalDevice, queueInfos.graphicsQueueIdx.value(), 0, &graphicsQueue);
	vkGetDeviceQueue(logicalDevice, queueInfos.presentQueueIdx.value(), 0, &presentQueue);
	vkGetDeviceQueue(logicalDevice, queueInfos.computeQueueIdx.value(), 0, &computeQueue);

	if (rayTracing) {
		auto load = [this](const char* name) { return vkGetDeviceProcAddr(logicalDevice, name); };
		rayTracingFunctions = {
			.createAccelerationStructure = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(load("vkCreateAccelerationStructureKHR")),
			.destroyAccelerationStructure = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(load("vkDestroyAccelerationStructureKHR")),
			.getBuildSizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(load("vkGetAccelerationStructureBuildSizesKHR")),
			.cmdBuild = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(load("vkCmdBuildAccelerationStructuresKHR")),
			.getAddress = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(load("vkGetAccelerationStructureDeviceAddressKHR")),
		};

		VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationProperties = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR
		};
		VkPhysicalDeviceProperties2 properties = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &accelerationProperties };
		vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
		scratchAlignment = std::max<VkDeviceSize>(accelerationProperties.minAccelerationStructureScratchOffsetAlignment, 1);
	}
}

foto::Device::~Device() {
	vkDestroyDevice(logicalDevice, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
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

VkSurfaceKHR foto::Device::createSurface(VkInstance instance, GLFWwindow* window) {
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	
	if (glfwCreateWindowSurface(instance, window , nullptr, &surface) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> createSurface()");
	}

	return surface;
}

VkPhysicalDevice foto::Device::selectPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {
	u32 deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
	if (deviceCount == 0) throw std::runtime_error("ERROR: vkDevice.cpp -> selectPhysicalDevice(): no Vulkan GPU found");
	
	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

	VkPhysicalDevice chosenDevice = VK_NULL_HANDLE;
	std::string rejected;
	for (auto device : devices) {
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(device, &properties);
		auto reject = [&](const std::string& reason) { rejected += std::string("\n  ") + properties.deviceName + ": " + reason; };
		if (properties.apiVersion < VK_API_VERSION_1_3) {
			reject("needs Vulkan 1.3, the driver reports " + std::to_string(VK_API_VERSION_MAJOR(properties.apiVersion)) + "." +
				   std::to_string(VK_API_VERSION_MINOR(properties.apiVersion)) + " (update the graphics driver)");
			continue;
		}

		// Check for Vulkan Features
		VkPhysicalDeviceVulkan12Features features12 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
		};
		VkPhysicalDeviceVulkan13Features features13 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
			.pNext = &features12
		};
		VkPhysicalDeviceFeatures2 features2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext = &features13
		};
		vkGetPhysicalDeviceFeatures2(device, &features2);
		if (!features13.dynamicRendering || !features13.synchronization2) {
			reject("no dynamic rendering or synchronization2");
			continue;
		}
		if (!features12.descriptorBindingPartiallyBound) {
			reject("no partially bound descriptors");
			continue;
		}
		if (!findQueueFamilies(device, surface).has_value()) {
			reject("no graphics, compute or present queue for this window");
			continue;
		}
		if (!extensionsSuitable(device, requiredDeviceExtensions)) {
			reject("no swapchain extension");
			continue;
		}
		SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device, surface);
		if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty()) {
			reject("can't present to this window");
			continue;
		}
		if (chooseTextureBinding(device).capacity < MIN_TEXTURES) {
			reject("can bind only " + std::to_string(chooseTextureBinding(device).capacity) + " textures per shader, needs " +
				   std::to_string(MIN_TEXTURES));
			continue;
		}

		chosenDevice = device;
		// Use discrete if found right way
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
	} 

	if (chosenDevice == VK_NULL_HANDLE) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> selectPhysicalDevice(): no usable GPU" + (rejected.empty() ? std::string(" found") : rejected));
	}
	return chosenDevice;
}

std::optional<foto::QueueInfos> foto::Device::findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
	u32 familyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
	std::vector<VkQueueFamilyProperties> properties(familyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, properties.data());

	std::optional<u32> graphics, present, compute;
	for (u32 i = 0; i < familyCount; i++) {
		if (!graphics && properties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) graphics = i;
		if (!compute && properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) compute = i;

		VkBool32 presentSupport = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
		if (!present && presentSupport) present = i;
	}

	if (!graphics || !compute || !present) return std::nullopt;
	return QueueInfos{
		.graphicsQueueIdx = graphics,
		.presentQueueIdx = present,
		.computeQueueIdx = compute
	};
}

bool foto::Device::extensionsSuitable(VkPhysicalDevice device, std::span<const char* const> required) {
	u32 deviceExtensionCount = 0;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &deviceExtensionCount, nullptr);

	std::vector<VkExtensionProperties> extensions(deviceExtensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &deviceExtensionCount, extensions.data());
	
	return std::all_of(required.begin(), required.end(),
		[&](const char* required) {
			return std::any_of(extensions.begin(), extensions.end(),
				[&](const VkExtensionProperties& e) {
					return std::strcmp(e.extensionName, required) == 0;
				});
		});
}

bool foto::Device::rayTracingSuitable(VkPhysicalDevice device) {
	if (!extensionsSuitable(device, rayTracingExtensions)) return false;

	VkPhysicalDeviceRayQueryFeaturesKHR rayQuery = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
	VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, .pNext = &rayQuery
	};
	VkPhysicalDeviceVulkan12Features features12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &acceleration };
	VkPhysicalDeviceFeatures2 features2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features12 };
	vkGetPhysicalDeviceFeatures2(device, &features2);

	return acceleration.accelerationStructure && rayQuery.rayQuery && features12.bufferDeviceAddress &&
		   features12.shaderSampledImageArrayNonUniformIndexing;
}

foto::TextureBinding foto::Device::chooseTextureBinding(VkPhysicalDevice device) {
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(device, &properties);
	const VkPhysicalDeviceLimits& limits = properties.limits;
	u32 regular = std::min({ limits.maxPerStageDescriptorSamplers, limits.maxPerStageDescriptorSampledImages,
							 limits.maxDescriptorSetSamplers, limits.maxDescriptorSetSampledImages });
	if (regular >= MAX_TEXTURES) return { MAX_TEXTURES, false };

	VkPhysicalDeviceVulkan12Properties properties12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES };
	VkPhysicalDeviceProperties2 properties2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &properties12 };
	vkGetPhysicalDeviceProperties2(device, &properties2);
	VkPhysicalDeviceVulkan12Features features12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	VkPhysicalDeviceFeatures2 features2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features12 };
	vkGetPhysicalDeviceFeatures2(device, &features2);

	if (features12.descriptorBindingSampledImageUpdateAfterBind) {
		u32 updateAfterBind = std::min({ properties12.maxPerStageDescriptorUpdateAfterBindSamplers,
										 properties12.maxPerStageDescriptorUpdateAfterBindSampledImages,
										 properties12.maxDescriptorSetUpdateAfterBindSamplers,
										 properties12.maxDescriptorSetUpdateAfterBindSampledImages });
		if (updateAfterBind > regular) return { std::min(updateAfterBind, MAX_TEXTURES), true };
	}
	return { regular, false };
}

VkDevice foto::Device::selectLogicalDevice(VkPhysicalDevice physicalDevice, const QueueInfos& queueInfo, bool rayTracing,
										  const TextureBinding& textureBinding) {
	VkDevice logicalDevice = VK_NULL_HANDLE;
	f32 priority = 1.0f;

	std::set<u32> uniqueFamilies = {
		queueInfo.graphicsQueueIdx.value(),
		queueInfo.presentQueueIdx.value(),
		queueInfo.computeQueueIdx.value()
	};

	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	for (auto family : uniqueFamilies) {
		VkDeviceQueueCreateInfo queueCreateInfo = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = family,
			.queueCount = 1,
			.pQueuePriorities = &priority
		};
		queueCreateInfos.push_back(queueCreateInfo);
	}

	VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
		.rayQuery = VK_TRUE
	};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
		.pNext = &rayQueryFeatures,
		.accelerationStructure = VK_TRUE
	};

	VkPhysicalDeviceVulkan12Features features12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		.pNext = rayTracing ? &accelerationFeatures : nullptr,
		.shaderSampledImageArrayNonUniformIndexing = rayTracing ? VK_TRUE : VK_FALSE,
		.descriptorBindingSampledImageUpdateAfterBind = textureBinding.updateAfterBind ? VK_TRUE : VK_FALSE,
		.descriptorBindingPartiallyBound = VK_TRUE,
		.bufferDeviceAddress = rayTracing ? VK_TRUE : VK_FALSE
	};

	std::vector<const char*> extensions = requiredDeviceExtensions;
	if (rayTracing) extensions.insert(extensions.end(), rayTracingExtensions.begin(), rayTracingExtensions.end());

	VkPhysicalDeviceVulkan13Features features13 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.pNext = &features12,
		.synchronization2 = VK_TRUE,
		.dynamicRendering = VK_TRUE
	};

	// Optional features: turn on whatever the GPU supports
	VkPhysicalDeviceFeatures supported;
	vkGetPhysicalDeviceFeatures(physicalDevice, &supported);

	VkPhysicalDeviceFeatures2 features2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		.pNext = &features13,
		.features = {
			.fillModeNonSolid = supported.fillModeNonSolid, // wireframe
			.samplerAnisotropy = supported.samplerAnisotropy
		}
	};

	VkDeviceCreateInfo createInfo = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pNext = &features2,
		.queueCreateInfoCount = static_cast<u32>(queueCreateInfos.size()),
		.pQueueCreateInfos = queueCreateInfos.data(),
		.enabledExtensionCount = static_cast<u32>(extensions.size()),
		.ppEnabledExtensionNames = extensions.data(),
		.pEnabledFeatures = nullptr

	};

	if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> selectLogicalDevice()");
	}

	return logicalDevice;
}

foto::u32 foto::Device::findMemoryType(u32 typeFilter, VkMemoryPropertyFlags properties) {
	// Available types of memory
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

	for (u32 i = 0; i < memProperties.memoryTypeCount; i++) {
		if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("ERROR: vkDevice.cpp -> findMemoryType()");
}

void foto::Device::immediateSubmit(VkCommandPool commandPool, const std::function<void(VkCommandBuffer)>& record) {
	VkCommandBufferAllocateInfo commandBufferAllocInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = commandPool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1
	};

	VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
	if (vkAllocateCommandBuffers(logicalDevice, &commandBufferAllocInfo, &commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> immediateSubmit() allocate");
	}

	VkCommandBufferBeginInfo beginInfo = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
	};

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> immediateSubmit() begin");
	}

	record(commandBuffer);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> immediateSubmit() end");
	}

	VkSubmitInfo submitInfo = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &commandBuffer
	};

	if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
		throw std::runtime_error("ERROR: vkDevice.cpp -> immediateSubmit() submit");
	}
	vkQueueWaitIdle(graphicsQueue);

	vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
}

foto::f32 foto::Device::getMaxSamplerAnisotropy() {
	VkPhysicalDeviceFeatures features;
	vkGetPhysicalDeviceFeatures(physicalDevice, &features);
	if (!features.samplerAnisotropy) return 1.0f;

	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(physicalDevice, &properties);
	return properties.limits.maxSamplerAnisotropy;
}

bool foto::Device::supportsWireframe() {
	VkPhysicalDeviceFeatures features;
	vkGetPhysicalDeviceFeatures(physicalDevice, &features);
	return features.fillModeNonSolid;
}

foto::SwapchainSupportDetails foto::Device::querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
	SwapchainSupportDetails swapchainSupport;

	u32 formatCount = 0;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
	if (formatCount > 0) {
		swapchainSupport.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, swapchainSupport.formats.data());
	} 

	u32 presentModeCount = 0;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
	if (presentModeCount > 0) {
		swapchainSupport.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, swapchainSupport.presentModes.data());
	}

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &swapchainSupport.capabilities);
	return swapchainSupport;
}
