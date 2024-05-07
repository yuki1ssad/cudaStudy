cmake_minimum_required(VERSION 3.16) # CMake最低版本号要求  
project(tensor) # 项目名字  
  
# 设置C++标准  
set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)  
  
# 添加头文件搜索路径  
include_directories(include)  
  
# 寻找./src下面所有.cpp为后缀的源文件，并且保存到SRC变量里面  
file(GLOB_RECURSE SRC ./src/*.cpp)  

# 编译SRC变量存储的源文件，编译生成目标文件命名为tensor  
#add_library(tensor SHARED ${SRC})
add_executable(tensor ${SRC})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tensor PUBLIC OpenMP::OpenMP_CXX)
endif()