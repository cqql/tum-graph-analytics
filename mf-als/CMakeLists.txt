add_executable(mf-als matrix_data.cpp mf-als.cpp worker.cpp)
target_link_libraries(mf-als ${gaml_LIBRARIES})
