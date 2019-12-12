cmake_minimum_required(VERSION 3.9)
project(VRVisor C CXX)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -g")

include_directories(include)

add_executable(live-cpu live.cpp capture.cpp effects.cpp kmeans.cpp kmeans-cpu.cpp)
target_link_libraries(live-cpu ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_executable(offline-cpu offline.cpp capture.cpp effects.cpp kmeans.cpp kmeans-cpu.cpp)
target_link_libraries(offline-cpu ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_executable(dual-cpu dual.cpp pipeline.cpp capture.cpp effects.cpp kmeans.cpp kmeans-cpu.cpp)
target_link_libraries(dual-cpu ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

find_package(CUDA QUIET)
if (CUDA_FOUND)
    set(CUDA_ARCH "53")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch sm_${CUDA_ARCH} -Xptxas=-v -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
    
    cuda_add_executable(live live.cpp capture.cpp effects.cpp kmeans.cpp kmeans.cu ${NVCC_FLAGS})
    target_link_libraries(live ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${CUDA_curand_LIBRARY})

    cuda_add_executable(offline offline.cpp capture.cpp effects.cpp kmeans.cpp kmeans.cu ${NVCC_FLAGS})
    target_link_libraries(offline ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${CUDA_curand_LIBRARY})

    cuda_add_executable(dual dual.cpp pipeline.cpp capture.cpp effects.cpp kmeans.cpp kmeans.cu ${NVCC_FLAGS})
    target_link_libraries(dual ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${CUDA_curand_LIBRARY})
endif (CUDA_FOUND)
