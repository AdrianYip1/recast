#include "imguiLayer.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <engine/platform/window.hpp>
#include <engine/rhi_vulkan/vkDevice.hpp>
#include <engine/rhi_vulkan/vkSwapchain.hpp>

static void checkVkResult(VkResult result) {
	if (result != VK_SUCCESS) {
		throw std::runtime_error("ERROR: imguiLayer.cpp -> Vulkan backend error " + std::to_string(result));
	}
}

foto::ImGuiLayer::ImGuiLayer(Window& window, Device& device, Swapchain& swapchain, VkFormat depthFormat) {
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.IniFilename = nullptr;
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForVulkan(window.getWindow(), true);

	VkFormat colorFormat = swapchain.getFormat();

	ImGui_ImplVulkan_InitInfo initInfo = {};
	initInfo.ApiVersion = VK_API_VERSION_1_3;
	initInfo.Instance = device.getInstance();
	initInfo.PhysicalDevice = device.getPhysicalDevice();
	initInfo.Device = device.getDevice();
	initInfo.QueueFamily = device.getGraphicsQueueFamily();
	initInfo.Queue = device.getGraphicsQ();
	initInfo.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE + 64;
	initInfo.MinImageCount = 2;
	initInfo.ImageCount = static_cast<u32>(swapchain.getImages().size());
	initInfo.UseDynamicRendering = true;
	initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 1,
		.pColorAttachmentFormats = &colorFormat,
		.depthAttachmentFormat = depthFormat,
	};
	initInfo.CheckVkResultFn = checkVkResult;

	if (!ImGui_ImplVulkan_Init(&initInfo)) {
		throw std::runtime_error("ERROR: imguiLayer.cpp -> ImGuiLayer(): ImGui_ImplVulkan_Init failed");
	}
}

foto::ImGuiLayer::~ImGuiLayer() {
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void foto::ImGuiLayer::beginFrame() {
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void foto::ImGuiLayer::endFrame() {
	ImGui::Render();
}

void foto::ImGuiLayer::record(VkCommandBuffer commandBuffer) {
	ImDrawData* drawData = ImGui::GetDrawData();
	if (drawData) {
		ImGui_ImplVulkan_RenderDrawData(drawData, commandBuffer);
	}
}

bool foto::ImGuiLayer::wantsMouse() {
	return ImGui::GetIO().WantCaptureMouse;
}

bool foto::ImGuiLayer::wantsKeyboard() {
	return ImGui::GetIO().WantCaptureKeyboard;
}
