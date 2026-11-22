#include "theme.hpp"

#include <cmath>
#include <filesystem>
#include <cstring>
#include <initializer_list>

#include <imgui_internal.h>
#include <stb_image.h>

namespace recast::ui {

namespace {

ImFont* title = nullptr;
ImFont* bold = nullptr;
ImTextureData logoTexture;

constexpr float FONT_SIZE = 16.0f;

// The first of these files that exists, or nullptr
ImFont* loadFirst(std::initializer_list<std::string> paths) {
	for (const std::string& path : paths) {
		std::error_code error;
		if (std::filesystem::exists(path, error)) return ImGui::GetIO().Fonts->AddFontFromFileTTF(path.c_str(), FONT_SIZE);
	}
	return nullptr;
}

// The screen is sRGB and ImGui's colours are taken as linear, so colours picked in sRGB (like the logo's) are
// converted back first or they come out washed out
float linear(int channel) {
	float v = channel / 255.0f;
	return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

ImVec4 rgb(int r, int g, int b, float a = 1.0f) {
	return ImVec4(linear(r), linear(g), linear(b), a);
}

void loadLogo(const std::string& path) {
	int width = 0, height = 0, channels = 0;
	unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
	if (!pixels) return;
	// same reason as rgb(): the image is sRGB, the screen re-encodes what ImGui draws
	for (int i = 0; i < width * height * 4; i++) {
		if (i % 4 != 3) pixels[i] = static_cast<unsigned char>(std::lround(255.0f * linear(pixels[i])));
	}
	logoTexture.Create(ImTextureFormat_RGBA32, width, height);
	std::memcpy(logoTexture.GetPixels(), pixels, static_cast<size_t>(width) * height * 4);
	logoTexture.UseColors = true;
	stbi_image_free(pixels);
	ImGui::RegisterUserTexture(&logoTexture);
}

} // namespace

ImTextureRef logo() {
	return logoTexture.GetTexRef();
}

ImVec2 logoSize() {
	return ImVec2(static_cast<float>(logoTexture.Width), static_cast<float>(logoTexture.Height));
}

ImVec4 purple() {
	return rgb(132, 96, 138);
}

ImVec4 paper() {
	return rgb(240, 231, 200);
}

ImVec4 paperLight() {
	return rgb(250, 245, 228);
}

ImFont* titleFont() {
	return title ? title : ImGui::GetFont();
}

ImFont* boldFont() {
	return bold ? bold : ImGui::GetFont();
}

bool accentButton(const char* label, ImVec2 size) {
	ImGui::PushStyleColor(ImGuiCol_Button, purple());
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, rgb(152, 114, 158));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, rgb(110, 78, 116));
	ImGui::PushStyleColor(ImGuiCol_Text, rgb(255, 255, 255));
	ImGui::PushFont(boldFont(), 0.0f);
	bool pressed = ImGui::Button(label, size);
	ImGui::PopFont();
	ImGui::PopStyleColor(4);
	return pressed;
}

void applyTheme(float dpiScale, const std::string& assetsDir) {
	if (dpiScale <= 0.0f) dpiScale = 1.0f;
	const std::string fonts = assetsDir + "/fonts/";

	// the first font added is the default; Segoe UI etc. if the bundled fonts are missing
	ImFont* regular = loadFirst({ fonts + "Montserrat-Light.ttf", "C:/Windows/Fonts/segoeuil.ttf",
								  "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" });
	if (!regular) ImGui::GetIO().Fonts->AddFontDefault();
	bold = loadFirst({ fonts + "Montserrat-SemiBold.ttf", "C:/Windows/Fonts/seguisb.ttf",
					   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" });
	// handwritten titles; Reenie Beanie draws small, so titles are pushed at about 1.4x the size of other fonts
	title = loadFirst({ fonts + "ReenieBeanie.ttf" });
	if (!title) title = bold;
	loadLogo(assetsDir + "/logo.png");

	ImGuiStyle& style = ImGui::GetStyle();
	style = ImGuiStyle();
	ImGui::StyleColorsLight(&style);

	style.WindowPadding = ImVec2(14.0f, 14.0f);
	style.FramePadding = ImVec2(8.0f, 5.0f);
	style.ItemSpacing = ImVec2(8.0f, 7.0f);
	style.WindowRounding = 10.0f;
	style.ChildRounding = 8.0f;
	style.FrameRounding = 5.0f;
	style.PopupRounding = 6.0f;
	style.GrabRounding = 5.0f;
	style.ScrollbarRounding = 5.0f;
	style.WindowBorderSize = 1.0f;
	style.ChildBorderSize = 1.0f;
	style.FrameBorderSize = 0.0f;
	style.SeparatorTextBorderSize = 1.0f;
	style.SeparatorTextPadding = ImVec2(0.0f, 4.0f);

	// light yellow paper, dark ink text, the logo's purple for everything that's picked or active
	const ImVec4 accent = purple();
	const ImVec4 accentSoft = rgb(132, 96, 138, 0.35f);
	const ImVec4 accentHover = rgb(152, 114, 158);
	const ImVec4 light = paperLight();
	const ImVec4 lighter = rgb(255, 252, 242);
	const ImVec4 dark = rgb(218, 206, 170);
	const ImVec4 ink = rgb(32, 26, 30);
	ImVec4* c = style.Colors;
	c[ImGuiCol_Text] = ink;
	c[ImGuiCol_TextDisabled] = rgb(112, 102, 92);
	c[ImGuiCol_WindowBg] = paper();
	c[ImGuiCol_ChildBg] = rgb(0, 0, 0, 0.0f);
	c[ImGuiCol_PopupBg] = light;
	c[ImGuiCol_Border] = rgb(32, 26, 30, 0.18f);
	c[ImGuiCol_FrameBg] = light;
	c[ImGuiCol_FrameBgHovered] = lighter;
	c[ImGuiCol_FrameBgActive] = lighter;
	c[ImGuiCol_TitleBg] = dark;
	c[ImGuiCol_TitleBgActive] = dark;
	c[ImGuiCol_TitleBgCollapsed] = dark;
	c[ImGuiCol_MenuBarBg] = paper();
	c[ImGuiCol_ScrollbarBg] = rgb(0, 0, 0, 0.0f);
	c[ImGuiCol_ScrollbarGrab] = dark;
	c[ImGuiCol_ScrollbarGrabHovered] = accentSoft;
	c[ImGuiCol_ScrollbarGrabActive] = accent;
	c[ImGuiCol_CheckMark] = accent;
	c[ImGuiCol_SliderGrab] = accent;
	c[ImGuiCol_SliderGrabActive] = accentHover;
	c[ImGuiCol_Button] = dark;
	c[ImGuiCol_ButtonHovered] = accentSoft;
	c[ImGuiCol_ButtonActive] = accentHover;
	c[ImGuiCol_Header] = accentSoft;
	c[ImGuiCol_HeaderHovered] = rgb(132, 96, 138, 0.25f);
	c[ImGuiCol_HeaderActive] = rgb(132, 96, 138, 0.5f);
	c[ImGuiCol_Separator] = rgb(32, 26, 30, 0.18f);
	c[ImGuiCol_SeparatorHovered] = accentHover;
	c[ImGuiCol_SeparatorActive] = accent;
	c[ImGuiCol_ResizeGrip] = rgb(0, 0, 0, 0.0f);
	c[ImGuiCol_ResizeGripHovered] = accentSoft;
	c[ImGuiCol_ResizeGripActive] = accent;
	c[ImGuiCol_TextSelectedBg] = accentSoft;
	c[ImGuiCol_NavCursor] = accent;
	c[ImGuiCol_ModalWindowDimBg] = rgb(32, 26, 30, 0.4f);

	// the right size on high-DPI laptop screens
	style.ScaleAllSizes(dpiScale);
	style.FontScaleDpi = dpiScale;
}

} // namespace recast::ui
