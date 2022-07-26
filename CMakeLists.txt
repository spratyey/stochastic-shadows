cmake_minimum_required(VERSION 3.12)
set(CMAKE_BUILD_TYPE_INIT "Debug")

project(OptixRenderer VERSION 0.1 LANGUAGES C CXX CUDA)

set(OptiX_ROOT_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0/")

set(owl_dir ${PROJECT_SOURCE_DIR}/owl/)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

include_directories("headers")
include_directories("cuda_headers")
include_directories("precomputation")

embed_ptx(
  OUTPUT_TARGET
    ptx_code
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    cuda/ltc_many_lights_cuda.cu
)

add_executable(ltc_many_lights 
# Host code
"ltc_many_lights.cpp" "scene.cpp" "model.cpp" 
# IMGUI files
"imgui.cpp" "imgui_demo.cpp" "imgui_draw.cpp" "imgui_impl_glfw.cpp" "imgui_impl_opengl2.cpp" "imgui_tables.cpp" 
"imgui_widgets.cpp"
)

add_executable(testbed "testbed.cpp")

target_link_libraries(ltc_many_lights 
PRIVATE
# Compiled PTX
ptx_code
# OWL libs
owl::owl owl_viewer)