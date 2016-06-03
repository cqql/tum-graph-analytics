cmake_minimum_required(VERSION 2.8)

project(gdesc CXX)

add_compile_options(-std=c++14)

add_subdirectory("../ps-lite" "ps-lite")
include_directories(${pslite_INCLUDE_DIR})

add_executable(gdesc gdesc.cpp)
target_link_libraries(gdesc pslite ${pslite_LINKER_LIBS})
