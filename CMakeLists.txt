cmake_minimum_required(VERSION 3.15)
project(foto VERSION 0.1.0 LANGUAGES CXX)

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
message(STATUS "GLSLC: ${Vulkan_GLSLC_EXECUTABLE}")

set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/engine/shaders)
file(GLOB SHADER_SOURCES CONFIGURE_DEPENDS
    ${SHADER_DIR}/*.vert
    ${SHADER_DIR}/*.frag
    ${SHADER_DIR}/*.comp)

# Shared code for #include "file.glsl": every shader recompiles when one of these changes
file(GLOB SHADER_INCLUDES CONFIGURE_DEPENDS ${SHADER_DIR}/*.glsl)

set(SPV_FILES "")
foreach(SHADER ${SHADER_SOURCES})
    add_custom_command(
        OUTPUT ${SHADER}.spv
        # vulkan1.3: SPIR-V 1.6, needed later for ray queries. -I: lets shaders #include from engine/shaders
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.3 -I ${SHADER_DIR} ${SHADER} -o ${SHADER}.spv
        DEPENDS ${SHADER} ${SHADER_INCLUDES}
        COMMENT "compiling ${SHADER}")
    list(APPEND SPV_FILES ${SHADER}.spv)
endforeach()

add_custom_target(shaders DEPENDS ${SPV_FILES})

foreach(SPV ${SPV_FILES})
    get_filename_component(SPV_NAME ${SPV} NAME)
    add_custom_command(TARGET shaders POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SPV} ${CMAKE_BINARY_DIR}/shaders/${SPV_NAME})
endforeach()
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE external/stb)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

add_library(foto_engine STATIC
    engine/platform/window.cpp
    engine/app/application.cpp
    engine/rhi_vulkan/vkDevice.cpp
    engine/rhi_vulkan/vkSwapchain.cpp 
    engine/rhi_vulkan/vkRender.cpp
    engine/rhi_vulkan/vkPipeline.cpp
    engine/rhi_vulkan/vkBuffer.cpp  "engine/rhi_vulkan/vkDescriptorHelper.cpp"
    engine/rhi_vulkan/vkImage.cpp
    engine/rhi_vulkan/vkMesh.cpp
    engine/scene/camera.cpp
    engine/assets/gltfLoader.cpp
    engine/ui/imguiLayer.cpp)

target_include_directories(foto_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(foto_engine PRIVATE external/cgltf)

target_link_libraries(foto_engine
    PUBLIC enginemath::enginemath imgui
    PRIVATE glfw Vulkan::Vulkan stb)

# Textures: copied next to the shaders so they load with the same relative paths
add_custom_target(assets
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets/textures ${CMAKE_BINARY_DIR}/textures
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets/models ${CMAKE_BINARY_DIR}/models
    COMMENT "copying assets")

add_executable(foto sandbox/main.cpp 
"sandbox/visualizer.cpp"
    sandbox/scene/sceneFile.cpp
    sandbox/scene/proxyMesh.cpp)
target_include_directories(foto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sandbox ${CMAKE_CURRENT_SOURCE_DIR}/external)

target_link_libraries(foto PRIVATE foto_engine glfw Vulkan::Vulkan stb)

add_dependencies(foto shaders assets)