#pragma once

#include <enginemath/enginemath.hpp>
#include <engine/core/defines.hpp>
#include <engine/core/stdIncludes.hpp>

struct GLFWwindow;

namespace foto {
	// Key/button codes are GLFW's (GLFW_KEY_W, GLFW_MOUSE_BUTTON_RIGHT, ...): #include <GLFW/glfw3.h>
	class Window {
	public:
		Window(u32 w, u32 h, const char* windowName);
		~Window();
		Window(const Window& other) = delete;
		Window& operator=(const Window& other) = delete;

		GLFWwindow* getWindow() { return window; };

		bool shouldClose();

		// Processes OS events and updates the per-frame input deltas below. Call once per frame.
		void pollEvents();

		// Changes framebufferResized
		void setFramebufferResized(bool info);
		bool getFramebufferResized() { return framebufferResized; };

		// Input: current state
		bool isKeyDown(int key);
		bool isMouseButtonDown(int button);
		enginemath::Vec2 getCursorPos();     // window pixels, (0,0) = top-left
		enginemath::Vec2 getFramebufferCursorPos(); // framebuffer pixels (differs from window pixels with display scaling)

		// Input: change since the previous pollEvents()
		enginemath::Vec2 getCursorDelta() { return cursorDelta; };
		f32 getScrollDelta() { return scrollDelta; };

		// Hides the cursor and locks it to the window, for mouse-look (deltas keep working)
		void setCursorCaptured(bool captured);
		bool isCursorCaptured() { return cursorCaptured; };

	private:
		GLFWwindow* window = nullptr;
		u32 WIDTH;
		u32 HEIGHT;
		const char* TITLE;
		bool framebufferResized = false;

		enginemath::Vec2 lastCursorPos;
		enginemath::Vec2 cursorDelta;
		f32 scrollDelta = 0.0f;
		f32 scrollAccumulated = 0.0f;
		bool hasLastCursorPos = false;
		bool cursorCaptured = false;

		void initWindow();

		// Sets framebufferResized to true when a resize is detected
		static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
		static void scrollCallback(GLFWwindow* window, double xOffset, double yOffset);
	};

} // namespace foto