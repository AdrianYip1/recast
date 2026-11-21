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
}

bool foto::Window::shouldClose() {
	return glfwWindowShouldClose(window);
}

void foto::Window::pollEvents() {
	glfwPollEvents();
}

void foto::Window::setFramebufferResized(bool info) {
	framebufferResized = info;
}

void foto::Window::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
	auto app = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
}
