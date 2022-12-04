cmake_minimum_required(VERSION 3.12)
set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(OptixRenderer VERSION 0.1 LANGUAGES C CXX CUDA)

set(OptiX_ROOT_DIR "/home/aakashkt/optix")
set(Optix_INCLUDE "/home/aakashkt/optix/include")

set(owl_dir ${PROJECT_SOURCE_DIR}/owl/)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

include_directories("headers")
include_directories("cuda_headers")
include_directories("precomputation")
include_directories("owl/3rdParty")

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

set(test_dir ${PROJECT_SOURCE_DIR}/test/)
add_subdirectory(${test_dir})

embed_ptx(
  OUTPUT_TARGET
    ptx_code_ltc_testbed
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    cuda/ltc_testbed.cu
)

add_executable(ltc_testbed
  "ltc_testbed.cpp"
  ${SRC_FILES}
)

add_executable(generate_bsp
  "generate_bsp.cpp"
  ${SRC_FILES}
)

target_link_libraries(ltc_testbed
    PRIVATE
    # compiled ptx
    ptx_code_ltc_testbed
    # OWL libs
    owl::owl owl_viewer
)

target_link_libraries(generate_bsp
    PRIVATE
    # OWL libs
    owl::owl owl_viewer
)