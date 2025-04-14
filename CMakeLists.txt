cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(CUDATutorial LANGUAGES CXX C CUDA)
find_package(CUDA 10.0 REQUIRED)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})


# set compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        ")
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

#[[
# reduce
add_executable(reduce_v0
    reduce/reduce_v0.cu
)
target_link_libraries(
    reduce_v0 PUBLIC
    -lcudart
    -lcudadevrt)
# reduce
add_executable(reduce_v1
    reduce/reduce_v1.cu
)
target_link_libraries(
    reduce_v1 PUBLIC
    -lcudart
    -lcudadevrt)
# reduce
add_executable(reduce_v2
    reduce/reduce_v2.cu
)
target_link_libraries(
    reduce_v2 PUBLIC
    -lcudart
    -lcudadevrt)
# reduce
add_executable(reduce_v3
    reduce/reduce_v3.cu
)
target_link_libraries(
    reduce_v3 PUBLIC
    -lcudart
    -lcudadevrt)

# reduce
add_executable(reduce_v4
    reduce/reduce_v4.cu
)
target_link_libraries(
    reduce_v4 PUBLIC
    -lcudart
    -lcudadevrt)

# reduce
add_executable(reduce_v4_2
    reduce/reduce_v4_2.cu
)
target_link_libraries(
    reduce_v4_2 PUBLIC
    -lcudart
    -lcudadevrt)

# reduce
add_executable(reduce_v5
    reduce/reduce_v5.cu
)
target_link_libraries(
    reduce_v5 PUBLIC
    -lcudart
    -lcudadevrt)

# reduce
add_executable(reduce_v6
    reduce/reduce_v6.cu
)
target_link_libraries(
    reduce_v6 PUBLIC
    -lcudart
    -lcudadevrt)

# reduce
add_executable(reduce_v7
    reduce/reduce_v7.cu
)
target_link_libraries(
    reduce_v7 PUBLIC
    -lcudart
    -lcudadevrt)

# reduce
add_executable(reduce_v8
    reduce/reduce_v8.cu
)
target_link_libraries(
    reduce_v8 PUBLIC
    -lcudart
    -lcudadevrt)]]


# sgemm
add_executable(sgemm_v0
    sgemm/sgemm_v0.cu
)
target_link_libraries(
    sgemm_v0 PUBLIC
    -lcudart
    -lcudadevrt)

add_executable(sgemm_v1
    sgemm/sgemm_v1.cu
)
target_link_libraries(
    sgemm_v1 PUBLIC
    -lcudart
    -lcudadevrt)

add_executable(sgemm_v2
    sgemm/sgemm_v2.cu
)
target_link_libraries(
    sgemm_v2 PUBLIC
    -lcudart
    -lcudadevrt)

add_executable(sgemm_v3
    sgemm/sgemm_v3.cu
)
target_link_libraries(
    sgemm_v3 PUBLIC
    -lcudart
    -lcudadevrt)

add_executable(sgemm_v4
    sgemm/sgemm_v4.cu
)
target_link_libraries(
    sgemm_v4 PUBLIC
    -lcudart
    -lcudadevrt)

add_executable(sgemm_v5
    sgemm/sgemm_v5.cu
)
target_link_libraries(
    sgemm_v5 PUBLIC
    -lcudart
    -lcudadevrt)

add_executable(sgemm_v6
    sgemm/sgemm_v6.cu
)
target_link_libraries(
    sgemm_v6 PUBLIC
    -lcudart
    -lcudadevrt)

add_executable(sgemm_v7
    sgemm/sgemm_v7.cu
)
target_link_libraries(
    sgemm_v7 PUBLIC
    -lcudart
    -lcudadevrt)


add_executable(add_profile
    kernel_profiling/add.cu
)
target_link_libraries(
    add_profile PUBLIC
    -lcudart
    -lcudadevrt)


add_executable(transpose_v1
    transpose/transpose_v1.cu
)
target_link_libraries(
    transpose_v1 PUBLIC
    -lcudart
    -lcudadevrt)


add_executable(transpose_v2
    transpose/transpose_v2.cu
)
target_link_libraries(
    transpose_v2 PUBLIC
    -lcudart
    -lcudadevrt)


