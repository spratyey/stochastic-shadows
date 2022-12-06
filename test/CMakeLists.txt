cmake_minimum_required(VERSION 3.12)

add_executable(test_bvh test_bvh.cu ${SRC_FILES})
set_property(TARGET test_bvh  PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test_bvh PROPERTY CUDA_ARCHITECTURES OFF)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_bvh PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
else()
    target_compile_options(test_bvh PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif ()

target_link_libraries(test_bvh
    PRIVATE
    # OWL libs
    owl::owl owl_viewer
)