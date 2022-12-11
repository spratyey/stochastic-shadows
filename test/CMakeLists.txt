cmake_minimum_required(VERSION 3.12)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_DEBUG} -DDISABLE_OPTIX")

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

add_executable(test_ltc test_ltc.cu ${SRC_FILES})
set_property(TARGET test_ltc  PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test_ltc PROPERTY CUDA_ARCHITECTURES OFF)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_ltc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
else()
    target_compile_options(test_ltc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif ()

target_link_libraries(test_ltc
    PRIVATE
    # OWL libs
    owl::owl owl_viewer
)