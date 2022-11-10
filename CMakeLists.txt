cmake_minimum_required(VERSION 3.12)
set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(OptixRenderer VERSION 0.1 LANGUAGES C CXX CUDA)

set(OptiX_ROOT_DIR "/home/cvit/ishaan/optix")
set(Optix_INCLUDE "/home/cvit/ishaan/optix/include")

set(owl_dir ${PROJECT_SOURCE_DIR}/owl/)
add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

include_directories("headers")
include_directories("cuda_headers")
include_directories("precomputation")
include_directories("owl/3rdParty")

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

embed_ptx(
  OUTPUT_TARGET
    ptx_code_ltc_many_lights
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    cuda/ltc_many_lights_cuda.cu
)

embed_ptx(
  OUTPUT_TARGET
    ptx_code_ltc_testbed
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    cuda/ltc_testbed.cu

    # ${CUDA_HEADERS}
)

add_executable(ltc_many_lights 
  "ltc_many_lights.cpp"
  ${SRC_FILES}
)

add_executable(ltc_testbed
  "ltc_testbed.cpp"
  ${SRC_FILES}
)

# add_executable(testbed "testbed.cpp")

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
