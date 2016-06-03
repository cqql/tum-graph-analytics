include ../project.mk

.PHONY: all
all: mf-als

.PHONY: clean
clean:
	rm mf-als mf-als.o

SOURCE_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp, %.o, $(SOURCE_FILES))

mf-als: $(OBJ_FILES)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) $(PETUUM_INCFLAGS) $(OBJ_FILES) $(UTIL_OBJ) \
	$(PETUUM_PS_LIB) $(PETUUM_ML_LIB) $(PETUUM_LDFLAGS) -lboost_program_options \
	-larmadillo -o $@

mf-als.o: mf-als.cpp $(BOSEN_LIBS)
	$(PETUUM_CXX) $(NDEBUG) $(PETUUM_CXXFLAGS) -Wno-unused-result \
	-Wno-deprecated-declarations $(PETUUM_INCFLAGS) -c mf-als.cpp -o $@

worker.o: worker.cpp
	$(PETUUM_CXX) $(NDEBUG) $(PETUUM_CXXFLAGS) -Wno-unused-result \
	-Wno-deprecated-declarations $(PETUUM_INCFLAGS) -c worker.cpp -o $@

matrix_data.o: matrix_data.cpp
	$(PETUUM_CXX) $(NDEBUG) $(PETUUM_CXXFLAGS) -Wno-unused-result \
	-Wno-deprecated-declarations $(PETUUM_INCFLAGS) -c matrix_data.cpp -o $@
