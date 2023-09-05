cmake_minimum_required(VERSION 3.12)
# set(CMAKE_BUILD_TYPE "Release") # 或者 "Release"
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()
set(CMAKE_CXX_STANDARD 17)
project(Voxelization)

find_package(Threads REQUIRED)

# 获取当前目录下所有的 .cpp 文件
file(GLOB CPP_SOURCES *.cpp)

# 添加所有的 .cpp 文件到可执行文件
add_executable(main ${CPP_SOURCES})
target_link_libraries(main PRIVATE Threads::Threads)
