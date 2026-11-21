#pragma once
#include <enginemath/enginemath.hpp>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

namespace foto {
	struct Ubo {
		alignas(16) enginemath::Mat4 model;
		alignas(16) enginemath::Mat4 view;
		alignas(16) enginemath::Mat4 proj;
	};
} // namespace foto