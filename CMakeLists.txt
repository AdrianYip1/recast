cmake_minimum_required(VERSION 3.15)
project(mr_poop VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external/enginemath)

# Forcing depth to be from 0 to 1 for Vulkan in enginemath
target_compile_definitions(enginemath INTERFACE ENGINEMATH_DEPTH_ZERO_TO_ONE)

# glfw: skip docs/tests/examples, only want the library
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

find_package(Vulkan REQUIRED)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE external/stb)

add_executable(mr_poop src/main.cpp)

target_link_libraries(mr_poop PRIVATE enginemath::enginemath glfw Vulkan::Vulkan stb)
