cmake_minimum_required(VERSION 2.8)

project(logreg CXX)

add_compile_options(-std=c++14)

add_subdirectory("../ps-lite" "ps-lite")
include_directories(${pslite_INCLUDE_DIR})

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(logreg ps_logreg_sgd.cpp)
target_link_libraries(logreg pslite ${pslite_LINKER_LIBS})
target_link_libraries(logreg ${ARMADILLO_LIBRARIES})
