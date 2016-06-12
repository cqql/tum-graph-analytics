cmake_minimum_required(VERSION 2.8)

project(gaml CXX)

# Load bosen CMake configuration
add_subdirectory(libs/bosen)

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

# Additional boost libraries that we commonly need
find_package(Boost COMPONENTS system program_options)

# Collect all libraries that we will link with almost every time
set(gaml_LIBRARIES petuum_bosen ${Bosen_LIBRARIES} ${Boost_LIBRARIES}
  ${ARMADILLO_LIBRARIES})

# Load our subprojects
add_subdirectory(mf-als)
