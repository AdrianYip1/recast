#include "theme.hpp"

#include <filesystem>
#include <initializer_list>

namespace recast::ui {

namespace {

ImFont* bold = nullptr;

constexpr float FONT_SIZE = 17.0f;

// The first of these that exists; ImGui's built-in pixel font if none does
ImFont* loadFirst(std::initializer_list<const char*> paths) {
	for (const char* path : paths) {
		std::error_code error;
		if (std::filesystem::exists(path, error)) return ImGui::GetIO().Fonts->AddFontFromFileTTF(path, FONT_SIZE);
	}
	return nullptr;
}

ImVec4 rgb(int r, int g, int b, float a = 1.0f) {
	return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a);
}

} // namespace

ImVec4 accentColor() {
	return rgb(64, 132, 246);
}

ImFont* boldFont() {
	return bold ? bold : ImGui::GetFont();
}

void applyTheme(float dpiScale) {
	if (dpiScale <= 0.0f) dpiScale = 1.0f;

	// a real UI font instead of ImGui's 13 px pixel font. The first one loaded is the default
	ImFont* regular = loadFirst({ "C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/SFNS.ttf",
								  "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
								  "/usr/share/fonts/TTF/DejaVuSans.ttf" });
	if (!regular) ImGui::GetIO().Fonts->AddFontDefault();
	bold = loadFirst({ "C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/segoeuib.ttf",
					   "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" });

	ImGuiStyle& style = ImGui::GetStyle();
	style = ImGuiStyle();
	ImGui::StyleColorsDark(&style);

	// roomier, rounded
	style.WindowPadding = ImVec2(14.0f, 14.0f);
	style.FramePadding = ImVec2(10.0f, 6.0f);
	style.ItemSpacing = ImVec2(8.0f, 8.0f);
	style.ItemInnerSpacing = ImVec2(6.0f, 6.0f);
	style.IndentSpacing = 16.0f;
	style.ScrollbarSize = 12.0f;
	style.GrabMinSize = 12.0f;
	style.WindowRounding = 10.0f;
	style.ChildRounding = 8.0f;
	style.FrameRounding = 6.0f;
	style.PopupRounding = 8.0f;
	style.ScrollbarRounding = 6.0f;
	style.GrabRounding = 6.0f;
	style.TabRounding = 6.0f;
	style.WindowBorderSize = 0.0f;
	style.FrameBorderSize = 0.0f;
	style.PopupBorderSize = 1.0f;
	style.WindowTitleAlign = ImVec2(0.0f, 0.5f);
	style.SeparatorTextBorderSize = 1.0f;
	style.SeparatorTextPadding = ImVec2(0.0f, 6.0f);

	// soft dark greys with one blue accent
	const ImVec4 accent = accentColor();
	const ImVec4 accentHover = rgb(92, 152, 250);
	const ImVec4 accentActive = rgb(44, 108, 220);
	ImVec4* c = style.Colors;
	c[ImGuiCol_Text] = rgb(232, 234, 238);
	c[ImGuiCol_TextDisabled] = rgb(140, 146, 156);
	c[ImGuiCol_WindowBg] = rgb(24, 26, 30, 0.94f);
	c[ImGuiCol_ChildBg] = rgb(30, 33, 38, 0.6f);
	c[ImGuiCol_PopupBg] = rgb(30, 33, 38, 0.98f);
	c[ImGuiCol_Border] = rgb(255, 255, 255, 0.08f);
	c[ImGuiCol_FrameBg] = rgb(40, 44, 51);
	c[ImGuiCol_FrameBgHovered] = rgb(50, 55, 64);
	c[ImGuiCol_FrameBgActive] = rgb(58, 64, 74);
	c[ImGuiCol_TitleBg] = rgb(24, 26, 30);
	c[ImGuiCol_TitleBgActive] = rgb(24, 26, 30);
	c[ImGuiCol_TitleBgCollapsed] = rgb(24, 26, 30, 0.8f);
	c[ImGuiCol_ScrollbarBg] = rgb(0, 0, 0, 0.0f);
	c[ImGuiCol_ScrollbarGrab] = rgb(70, 76, 86);
	c[ImGuiCol_ScrollbarGrabHovered] = rgb(86, 92, 104);
	c[ImGuiCol_ScrollbarGrabActive] = rgb(100, 108, 120);
	c[ImGuiCol_CheckMark] = accent;
	c[ImGuiCol_SliderGrab] = accent;
	c[ImGuiCol_SliderGrabActive] = accentHover;
	c[ImGuiCol_Button] = rgb(48, 53, 61);
	c[ImGuiCol_ButtonHovered] = rgb(62, 68, 78);
	c[ImGuiCol_ButtonActive] = rgb(74, 81, 93);
	c[ImGuiCol_Header] = rgb(40, 44, 51);
	c[ImGuiCol_HeaderHovered] = rgb(50, 55, 64);
	c[ImGuiCol_HeaderActive] = rgb(58, 64, 74);
	c[ImGuiCol_Separator] = rgb(255, 255, 255, 0.08f);
	c[ImGuiCol_SeparatorHovered] = accentHover;
	c[ImGuiCol_SeparatorActive] = accentActive;
	c[ImGuiCol_ResizeGrip] = rgb(0, 0, 0, 0.0f);
	c[ImGuiCol_ResizeGripHovered] = accentHover;
	c[ImGuiCol_ResizeGripActive] = accentActive;
	c[ImGuiCol_Tab] = rgb(40, 44, 51);
	c[ImGuiCol_TabHovered] = rgb(62, 68, 78);
	c[ImGuiCol_TabSelected] = rgb(54, 60, 70);
	c[ImGuiCol_TextSelectedBg] = rgb(64, 132, 246, 0.35f);
	c[ImGuiCol_NavCursor] = accent;
	c[ImGuiCol_ModalWindowDimBg] = rgb(0, 0, 0, 0.5f);

	// sharp and the right size on high-DPI laptop screens
	style.ScaleAllSizes(dpiScale);
	style.FontScaleDpi = dpiScale;
}

} // namespace recast::ui
