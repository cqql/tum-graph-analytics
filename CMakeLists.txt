cmake_minimum_required(VERSION 2.8)

project(gaml CXX)

add_definitions(-std=c++11)

# Load bosen CMake configuration
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libs/bosen)

# Include headers from boost and dependencies
include_directories(${Bosen_INCLUDE_DIRS} ${third_INCLUDE_DIRS}
	${ZeroMQ_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS} ${YAML_INCLUDE_DIRS}
	${GPERFTOOLS_INCLUDE_DIRS} ${SNAPPY_INCLUDE_DIRS} ${SPARSEHASH_INCLUDE_DIR}
	${LevelDB_INCLUDE_PATH} ${GFLAGS_INCLUDE_DIRS} ${LIBCONFIG_INCLUDE_DIR}
	${Tcmalloc_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})

# Collect all bosen libraries in one list
set(Bosen_LIBRARIES rt nsl pthread ${ZeroMQ_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}
  ${GLOG_LIBRARY} ${Tcmalloc_LIBRARIES} ${LIBCONFIG_LIBRARIES}
  ${SNAPPY_LIBRARIES} ${Boost_LIBRARIES} ${YAML_LIBRARIES} ${LevelDB_LIBRARY}
  ${GFLAGS_LIBRARIES})

# Initialize armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Initialize HDF5
find_package(HDF5 REQUIRED CXX)
include_directories(${HDF5_INCLUDE_DIR})

# Additional boost libraries that we commonly need
find_package(Boost COMPONENTS system program_options)

# Collect all libraries that we will link with almost every time
set(gaml_LIBRARIES petuum_bosen ${Bosen_LIBRARIES} ${Boost_LIBRARIES}
  ${ARMADILLO_LIBRARIES} ${HDF5_LIBRARIES})

# List all source files
set(gaml_SRCS gaml/io/matrix.cpp gaml/io/matrix_slice.cpp gaml/mf/worker.cpp
gaml/util/table.cpp)

set(tf_SRCS ${gaml_SRCS} gaml/io/tensor.cpp gaml/tf/worker.cpp)

set(als_SRCS ${gaml_SRCS} gaml/mf/als/worker.cpp
  gaml/mf/als/pseudo_inverse_solver.cpp gaml/mf/als/ridge_regression_solver.cpp
  gaml/mf/als/logger.cpp)

set(biases_SRCS ${gaml_SRCS} gaml/biases/worker.cpp ${als_SRCS})

set(gd_SRCS ${gaml_SRCS} gaml/mf/gd/worker.cpp gaml/mf/gd/r_projection.cpp
  gaml/mf/gd/nn_projection.cpp)

set(koren_SRCS ${gaml_SRCS} gaml/mf/koren/worker.cpp)

set(softmax_srcs
  gaml/softmax/src/common.hpp
  gaml/softmax/src/engine.hpp gaml/softmax/src/engine.cpp
  gaml/softmax/src/sgd_solver.hpp gaml/softmax/src/sgd_solver.cpp)

add_executable(mf-als gaml/bins/mf-als.cpp ${als_SRCS})
target_link_libraries(mf-als ${gaml_LIBRARIES})

add_executable(bias-mf gaml/bins/bias-mf.cpp ${biases_SRCS})
target_link_libraries(bias-mf ${gaml_LIBRARIES})

add_executable(softmax gaml/softmax/src/softmax.cpp ${softmax_srcs})
target_link_libraries(softmax ${gaml_LIBRARIES})
add_custom_command(TARGET softmax
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:softmax> gaml/softmax/bin)

add_executable(tf gaml/bins/tf.cpp ${tf_SRCS})
target_link_libraries(tf ${gaml_LIBRARIES})

add_executable(koren gaml/bins/koren.cpp ${koren_SRCS})
target_link_libraries(koren ${gaml_LIBRARIES})
