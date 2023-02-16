cmake_minimum_required(VERSION 3.20)
project(reaction_diffusion)

include(FetchContent)

##########################################
# Threads are found automagically
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

################################################################
# OneTBB
# TBB can be downloaded from https://github.com/oneapi-src/oneTBB/releases/download/v2021.4.0/oneapi-tbb-2021.4.0-win.zip
set(TBB_DIR "c:/dev-c/oneapi-tbb-2021.7.0/lib/cmake/tbb")
find_package(TBB REQUIRED)
################################################################
### Use OpenCV from https://opencv.org/releases/
### note that it's self-extracting archive that can be opened in 7zFM and opencv/build
### directory extracted to e.g. e:/dev-c/opencv-4.5.3
set(OpenCV_DIR "c:/dev-c/opencv-4.7.0/build")
find_package(OpenCV 4.2 REQUIRED)
set_target_properties( ${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE )

################################################################
# FMT (c++20 adds majority off this as std::format)
set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 9.1.0
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(fmt)

################################################################
# NOTE: TO SET CLANG COMPILER (OPTIONAL)
# export CC=/usr/bin/clang; CXX=/usr/bin/clang++
# or apt install llvm-12 clang-12 lld-12
#    export CC=/usr/bin/clang-12; CXX=/usr/bin/clang++-12
#    cmake . -D_CMAKE_TOOLCHAIN_PREFIX=llvm- -Bbuild-linux
# not sure how to make clang-12 and toolchain v12 default,
# there are some symlinks
################################################################


################################################################
### SET C++17 STANDARD
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Native arch on Linux (do not distribute such binary, it's for optimizations)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

### SET WARNINGS
if (MSVC)
    # Enable all, but 
    # warning C4127: conditional expression is constant (https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-4-c4127?view=msvc-160)
    add_compile_options(/W4 /wd4127)
else()
	# march native for avx	
    add_compile_options(-Wall -Wextra -pedantic)
endif()

add_executable(reaction_diffusion sources/reaction_diffusion.cpp)
target_include_directories(reaction_diffusion PUBLIC include)
target_link_libraries(reaction_diffusion 
    PRIVATE 
        Threads::Threads
        TBB::tbb
        fmt::fmt
        ${OpenCV_LIBS})

# TODO: unresolved problem: no idea how to recognize debug and release
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

# TODO: copy DLLS possibly as a post-build step
# OpenCV has interesting variables
#_OpenCV_LIB_PATH=C:\dev-c\opencv-4.7.0\build\x64\vc16\bin
#__location_dbg=C:/dev-c/opencv-4.7.0/build/x64/vc16/bin/opencv_world470d.dll
#__location_release=C:/dev-c/opencv-4.7.0/build/x64/vc16/bin/opencv_world470.dll
