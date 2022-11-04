cmake_minimum_required(VERSION 3.12)
set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(OptixRenderer VERSION 0.1 LANGUAGES C CXX CUDA)

set(OptiX_ROOT_DIR "/home/aakashkt/optix")

set(owl_dir ${PROJECT_SOURCE_DIR}/owl/)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

include_directories("headers")
include_directories("precomputation")

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB CUDA_HEADERS ${PROJECT_SOURCE_DIR}/headers/cuda_headers/**/*.cuh)

embed_ptx(
  OUTPUT_TARGET
    ptx_code_ltc_many_lights
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    cuda/ltc_many_lights_cuda.cu
    
    headers/cuda_headers/bvh.cuh
    headers/cuda_headers/common.cuh
    headers/cuda_headers/frostbite.cuh
    headers/cuda_headers/frostbite_anisotropic.cuh
    headers/cuda_headers/hit.cuh
    headers/cuda_headers/ltc_utils.cuh
    headers/cuda_headers/polygon_utils.cuh
    headers/cuda_headers/utils.cuh
)

embed_ptx(
  OUTPUT_TARGET
    ptx_code_ltc_testbed
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    cuda/ltc_testbed.cu
    
    ${CUDA_HEADERS}
)

add_executable(ltc_many_lights 
    # Host code
    "ltc_many_lights.cpp" "scene.cpp" "model.cpp"    # IMGUI files
    "imgui.cpp" "imgui_demo.cpp" "imgui_draw.cpp" "imgui_impl_glfw.cpp" "imgui_impl_opengl2.cpp" "imgui_tables.cpp" 
    "imgui_widgets.cpp"
)

add_executable(ltc_testbed
    # Host code
    "ltc_testbed.cpp" ${SRC_FILES}
 
    # IMGUI files
    "imgui.cpp" "imgui_demo.cpp" "imgui_draw.cpp" "imgui_impl_glfw.cpp" "imgui_impl_opengl2.cpp" "imgui_tables.cpp" 
    "imgui_widgets.cpp"
)

add_executable(testbed "testbed.cpp")

target_link_libraries(ltc_many_lights 
    PRIVATE
    # Compiled PTX
    ptx_code_ltc_many_lights
    # OWL libs
    owl::owl owl_viewer
 )

target_link_libraries(ltc_testbed
    PRIVATE
    # Compiled PTX
    ptx_code_ltc_testbed
    # OWL libs
    owl::owl owl_viewer
)
