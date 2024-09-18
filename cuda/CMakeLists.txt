cmake_minimum_required(VERSION 3.5.0)
project(OpenCV_GPU VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(OpenCV_DIR "C:/opencv_gpu/build/install")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if (CUDA_FOUND)
    set (CUDA_NVCC_FLAGS "
    -ccbin C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64/cl.exe;
    -gencode;arch=compute_50,code=sm_50;  
    -gencode;arch=compute_52,code=sm_52;
    -gencode;arch=compute_61,code=sm_61;
    -gencode;arch=compute_70,code=sm_70; 
    -gencode;arch=compute_75,code=sm_75;")

    if (UNIX)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
        set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
    endif (UNIX)

    if (WIN32)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler /Wextra")
    endif (WIN32)

    # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
else (CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

set(SOURCES
    main.cpp
    utils.cpp
    simcc_decoder.cu
)

add_executable(OpenCV_GPU ${SOURCES})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")

target_link_libraries(OpenCV_GPU ${OpenCV_LIBS})

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)