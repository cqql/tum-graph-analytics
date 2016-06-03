cmake_minimum_required(VERSION 2.8)

project(mf-als CXX)

add_subdirectory("../ps-lite" "ps-lite")
include_directories(${pslite_INCLUDE_DIR})

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(mf-als mf-als.cpp)
set_property(TARGET mf-als PROPERTY CXX_STANDARD 14)
target_link_libraries(mf-als
  pslite ${pslite_LINKER_LIBS} ${ARMADILLO_LIBRARIES})
