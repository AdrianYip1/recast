#include "theme.hpp"

#include <cmath>
#include <filesystem>
#include <initializer_list>

namespace recast::ui {

namespace {

ImFont* bold = nullptr;

constexpr float FONT_SIZE = 16.0f;

// The first of these that exists, or nullptr
ImFont* loadFirst(std::initializer_list<const char*> paths) {
	for (const char* path : paths) {
		std::error_code error;
		if (std::filesystem::exists(path, error)) return ImGui::GetIO().Fonts->AddFontFromFileTTF(path, FONT_SIZE);
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

} // namespace

ImVec4 purple() {
	return rgb(132, 96, 138);
}

ImVec4 beige() {
	return rgb(208, 198, 180);
}

ImFont* boldFont() {
	return bold ? bold : ImGui::GetFont();
}

bool accentButton(const char* label, float width) {
	ImGui::PushStyleColor(ImGuiCol_Button, purple());
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, rgb(152, 114, 158));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, rgb(110, 78, 116));
	ImGui::PushStyleColor(ImGuiCol_Text, rgb(255, 255, 255));
	bool pressed = ImGui::Button(label, ImVec2(width, 0.0f));
	ImGui::PopStyleColor(4);
	return pressed;
}

void applyTheme(float dpiScale) {
	if (dpiScale <= 0.0f) dpiScale = 1.0f;

	// a UI font instead of ImGui's 13 px pixel font; the first one added is the default
	ImFont* regular = loadFirst({ "C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Helvetica.ttc",
								  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf" });
	if (!regular) ImGui::GetIO().Fonts->AddFontDefault();
	bold = loadFirst({ "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/seguisb.ttf",
					   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" });

	ImGuiStyle& style = ImGui::GetStyle();
	style = ImGuiStyle();
	ImGui::StyleColorsLight(&style);

	style.WindowPadding = ImVec2(12.0f, 12.0f);
	style.FramePadding = ImVec2(8.0f, 4.0f);
	style.ItemSpacing = ImVec2(8.0f, 6.0f);
	style.WindowRounding = 8.0f;
	style.ChildRounding = 6.0f;
	style.FrameRounding = 5.0f;
	style.PopupRounding = 6.0f;
	style.GrabRounding = 5.0f;
	style.ScrollbarRounding = 5.0f;
	style.WindowBorderSize = 1.0f;
	style.FrameBorderSize = 0.0f;

	// the logo's beige paper, dark ink text and its purple for everything that's picked or active
	const ImVec4 accent = purple();
	const ImVec4 accentSoft = rgb(132, 96, 138, 0.35f);
	const ImVec4 accentHover = rgb(152, 114, 158);
	const ImVec4 paper = beige();
	const ImVec4 paperLight = rgb(228, 220, 205);
	const ImVec4 paperLighter = rgb(238, 232, 220);
	const ImVec4 paperDark = rgb(190, 178, 158);
	const ImVec4 ink = rgb(32, 26, 30);
	ImVec4* c = style.Colors;
	c[ImGuiCol_Text] = ink;
	c[ImGuiCol_TextDisabled] = rgb(112, 102, 96);
	c[ImGuiCol_WindowBg] = rgb(208, 198, 180, 0.97f);
	c[ImGuiCol_ChildBg] = rgb(0, 0, 0, 0.0f);
	c[ImGuiCol_PopupBg] = paperLight;
	c[ImGuiCol_Border] = rgb(32, 26, 30, 0.25f);
	c[ImGuiCol_FrameBg] = paperLight;
	c[ImGuiCol_FrameBgHovered] = paperLighter;
	c[ImGuiCol_FrameBgActive] = paperLighter;
	c[ImGuiCol_TitleBg] = paperDark;
	c[ImGuiCol_TitleBgActive] = paperDark;
	c[ImGuiCol_TitleBgCollapsed] = paperDark;
	c[ImGuiCol_MenuBarBg] = paper;
	c[ImGuiCol_ScrollbarBg] = rgb(0, 0, 0, 0.0f);
	c[ImGuiCol_ScrollbarGrab] = paperDark;
	c[ImGuiCol_ScrollbarGrabHovered] = accentSoft;
	c[ImGuiCol_ScrollbarGrabActive] = accent;
	c[ImGuiCol_CheckMark] = accent;
	c[ImGuiCol_SliderGrab] = accent;
	c[ImGuiCol_SliderGrabActive] = accentHover;
	c[ImGuiCol_Button] = paperDark;
	c[ImGuiCol_ButtonHovered] = accentSoft;
	c[ImGuiCol_ButtonActive] = accentHover;
	c[ImGuiCol_Header] = accentSoft;
	c[ImGuiCol_HeaderHovered] = rgb(132, 96, 138, 0.25f);
	c[ImGuiCol_HeaderActive] = rgb(132, 96, 138, 0.5f);
	c[ImGuiCol_Separator] = rgb(32, 26, 30, 0.2f);
	c[ImGuiCol_SeparatorHovered] = accentHover;
	c[ImGuiCol_SeparatorActive] = accent;
	c[ImGuiCol_ResizeGrip] = rgb(132, 96, 138, 0.2f);
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
