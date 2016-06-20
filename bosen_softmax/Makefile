SOFTMAX_DIR := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)

# path to bosen is needed
# could be pased also as arg, e.g. "make PETUUM_ROOT=/home/rosko/petuum_test/bosen"
PETUUM_ROOT = /

include $(PETUUM_ROOT)/defns.mk

SOFTMAX_SRC = $(wildcard $(SOFTMAX_DIR)/src/*.cpp)
SOFTMAX_HDR = $(wildcard $(SOFTMAX_DIR)/src/*.hpp)
SOFTMAX_BIN = $(SOFTMAX_DIR)/bin
SOFTMAX_OBJ = $(SOFTMAX_SRC:.cpp=.o)
NDEBUG = -DNDEBUG
CFLAGS = -g

all: $(SOFTMAX_BIN)/softmax_main

$(SOFTMAX_BIN):
	mkdir -p $(SOFTMAX_BIN)

$(SOFTMAX_BIN)/softmax_main: $(SOFTMAX_OBJ) $(PETUUM_PS_LIB) $(PETUUM_ML_LIB) $(SOFTMAX_BIN)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) $(PETUUM_INCFLAGS) \
	$(SOFTMAX_OBJ) $(PETUUM_PS_LIB) $(PETUUM_ML_LIB) $(PETUUM_LDFLAGS) -o $@

$(SOFTMAX_OBJ): %.o: %.cpp $(SOFTMAX_HDR)
	$(PETUUM_CXX) $(NDEBUG) $(PETUUM_CXXFLAGS) -Wno-unused-result \
		$(PETUUM_INCFLAGS) -c $< -o $@

$(SOFTMAX_BIN)/gen_data_sparse: $(SOFTMAX_DIR)/src/tools/gen_data_sparse.cpp $(SOFTMAX_BIN)
	$(PETUUM_CXX) $(PETUUM_CXXFLAGS) $(PETUUM_INCFLAGS) \
	$< $(PETUUM_PS_LIB) $(PETUUM_LDFLAGS) -o $@

clean:
	rm -rf $(SOFTMAX_OBJ)
	rm -rf $(SOFTMAX_BIN)

.PHONY: clean
