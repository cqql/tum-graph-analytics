cmake_minimum_required(VERSION 2.8)

project(linalg CXX)

add_compile_options(-std=c++14)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")
# Ignore deprecated C++ features in Eigen
add_compile_options(-Wno-deprecated-declarations)

add_executable(eigen eigen.cpp)
add_executable(armadillo armadillo.cpp)

target_link_libraries(armadillo ${ARMADILLO_LIBRARIES})
