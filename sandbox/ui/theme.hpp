#pragma once

#include <imgui.h>

namespace recast::ui {

// Fonts, spacing and colours for the whole panel. Call once after ImGui is created, before the first frame.
// dpiScale is the monitor's content scale (1.0 at 100%, 1.5 at 150%...)
void applyTheme(float dpiScale);

// The bold font loaded by applyTheme (the regular font if no bold one was found), for headings
ImFont* boldFont();

// The accent colour, for buttons that start a main step
ImVec4 accentColor();

} // namespace recast::ui
