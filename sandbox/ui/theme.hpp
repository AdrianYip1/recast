#pragma once

#include <imgui.h>

namespace recast::ui {

// Colours from the ReCast logo: its purple and the beige paper behind it
ImVec4 purple();
ImVec4 beige();

// Fonts, spacing and colours for the whole UI. Call once after ImGui is created, before the first frame.
// dpiScale is the monitor's content scale (1.0 at 100%, 1.5 at 150%...)
void applyTheme(float dpiScale);

// The bold font loaded by applyTheme (the regular font if no bold one was found), for titles
ImFont* boldFont();

// A purple button with white text, for the main action on screen. width 0 = fit the label
bool accentButton(const char* label, float width = 0.0f);

} // namespace recast::ui
