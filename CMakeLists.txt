cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(CUDATutorial LANGUAGES CXX C CUDA)
find_package(CUDA 10.0 REQUIRED)

set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})




# reduce
add_executable(run_reduce
    reduce/reduce_v0.cu
)
target_link_libraries(
    run_reduce PUBLIC
    -lcudart
    -lcudadevrt)

add_executable(run_reduce_1
    reduce/reduce_v1.cu
)
target_link_libraries(
    run_reduce_1 PUBLIC
    -lcudart
    -lcudadevrt)
add_executable(run_reduce_2
    reduce/reduce_v1.cu
)
target_link_libraries(
    run_reduce_2 PUBLIC
    -lcudart
    -lcudadevrt)
