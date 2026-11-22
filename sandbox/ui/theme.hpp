#pragma once

#include <string>

#include <imgui.h>

namespace recast::ui {

// Colours from the ReCast logo: its purple, and a light yellow paper for backgrounds
ImVec4 purple();
ImVec4 paper();
ImVec4 paperLight(); // cards and fields on the paper

// Fonts (Source Sans Pro for text, Source Serif Pro Bold for titles, from assetsDir/fonts), spacing and colours for
// the whole UI. Call once after ImGui is created, before the first frame. dpiScale is the monitor's content scale
void applyTheme(float dpiScale, const std::string& assetsDir);

ImFont* titleFont(); // Source Serif Pro Bold
ImFont* boldFont();  // Source Sans Pro Semibold

// A purple button with white text, for the main action on screen. A size of 0 fits the label
bool accentButton(const char* label, ImVec2 size = ImVec2(0.0f, 0.0f));

// The ReCast logo (assetsDir/logo.png); size is (0, 0) when it couldn't be loaded
ImTextureRef logo();
ImVec2 logoSize();

} // namespace recast::ui
