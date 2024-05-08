# 指定CMake的最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称和版本
project(DeepLearningFramework VERSION 1.0)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 包含CUDA
find_package(CUDA REQUIRED)

# 包含GTest
enable_testing()
find_package(GTest REQUIRED)

# 包含头文件目录
include_directories(include)

# 定义源文件和测试源文件
set(SOURCES
    src/matrix_multiply.cpp
)

set(TEST_SOURCES
    test/test_matrix_multiply.cpp
)

# 如果有CUDA文件，添加相应的库
if(CUDA_FOUND)
    set(SOURCES ${SOURCES} src/matrix_multiply.cu)
    cuda_include_directories(${PROJECT_SOURCE_DIR}/include)
    cuda_add_library(matrix_lib ${SOURCES})
else()
    add_library(matrix_lib ${SOURCES})
endif()

# 添加可执行文件，链接到相应的库
add_executable(matrix_multiply_executable src/matrix_multiply_main.cpp)
target_link_libraries(matrix_multiply_executable matrix_lib)

add_executable(test_matrix_multiply ${TEST_SOURCES})
target_link_libraries(test_matrix_multiply matrix_lib GTest::GTest GTest::Main)

# 如果CUDA存在，链接CUDA的库
if(CUDA_FOUND)
    target_link_libraries(test_matrix_multiply ${CUDA_LIBRARIES})
endif()

# 添加测试
add_test(NAME MatrixMultiplyTest COMMAND test_matrix_multiply)