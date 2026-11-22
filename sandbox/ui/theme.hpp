#pragma once

#include <string>

#include <imgui.h>

namespace recast::ui {

// Colours from the ReCast logo: its purple, and a light yellow paper for backgrounds
ImVec4 purple();
ImVec4 paper();
ImVec4 paperLight(); // cards and fields on the paper

// Fonts (Montserrat Light for text, Pinecone for titles, from assetsDir/fonts), spacing and colours for the whole UI.
// Call once after ImGui is created, before the first frame. dpiScale is the monitor's content scale
void applyTheme(float dpiScale, const std::string& assetsDir);

ImFont* titleFont(); // Pinecone (Montserrat SemiBold until its file is added)
ImFont* boldFont();  // Montserrat SemiBold

// A purple button with white text, for the main action on screen. A size of 0 fits the label
bool accentButton(const char* label, ImVec2 size = ImVec2(0.0f, 0.0f));

} // namespace recast::ui
