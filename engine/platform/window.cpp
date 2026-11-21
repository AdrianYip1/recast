#include <GLFW/glfw3.h>
#include "window.hpp"

foto::Window::Window(u32 w, u32 h, const char* windowName) :
	WIDTH(w), HEIGHT(h), TITLE(windowName)
{
	initWindow();
}

foto::Window::~Window() {
	glfwDestroyWindow(window);
	glfwTerminate();
}

void foto::Window::initWindow() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, nullptr, nullptr);

	if (!window) {
		throw std::runtime_error("ERROR: window.cpp -> initWindow()");
	}

	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	glfwSetScrollCallback(window, scrollCallback);
}

bool foto::Window::shouldClose() {
	return glfwWindowShouldClose(window);
}

void foto::Window::pollEvents() {
	glfwPollEvents();

	// Scroll arrives through the callback during glfwPollEvents
	scrollDelta = scrollAccumulated;
	scrollAccumulated = 0.0f;

	enginemath::Vec2 cursor = getCursorPos();
	cursorDelta = hasLastCursorPos ? cursor - lastCursorPos : enginemath::Vec2();
	lastCursorPos = cursor;
	hasLastCursorPos = true;
}

void foto::Window::setFramebufferResized(bool info) {
	framebufferResized = info;
}

bool foto::Window::isKeyDown(int key) {
	return glfwGetKey(window, key) == GLFW_PRESS;
}

bool foto::Window::isMouseButtonDown(int button) {
	return glfwGetMouseButton(window, button) == GLFW_PRESS;
}

enginemath::Vec2 foto::Window::getCursorPos() {
	double x = 0.0, y = 0.0;
	glfwGetCursorPos(window, &x, &y);
	return enginemath::Vec2(static_cast<f32>(x), static_cast<f32>(y));
}

void foto::Window::setCursorCaptured(bool captured) {
	if (captured == cursorCaptured) return;
	cursorCaptured = captured;

	glfwSetInputMode(window, GLFW_CURSOR, captured ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
	if (glfwRawMouseMotionSupported()) {
		glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, captured ? GLFW_TRUE : GLFW_FALSE);
	}

	// The cursor jumps when the mode changes: don't report that as movement
	hasLastCursorPos = false;
}

void foto::Window::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
	auto app = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
}

void foto::Window::scrollCallback(GLFWwindow* window, double xOffset, double yOffset) {
	auto app = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
	app->scrollAccumulated += static_cast<f32>(yOffset);
}
