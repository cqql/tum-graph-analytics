# URLs
BOSEN_REPO := "https://github.com/petuum/bosen.git"
BOSEN_THIRD_PARTY_REPO := "https://github.com/petuum/third_party.git"
STRADS_REPO := "https://github.com/petuum/strads.git"

# Petuum installation path
PROJECT_ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
LIBS_DIR := $(PROJECT_ROOT)/libs
BOSEN_ROOT := $(LIBS_DIR)/bosen
BOSEN_THIRD_PARTY_ROOT := $(LIBS_DIR)/bosen-third-party
STRADS_ROOT := $(LIBS_DIR)/strads

BOSEN_THIRD_PARTY_LIBS := $(addprefix $(BOSEN_THIRD_PARTY_ROOT)/lib/,\
libboost_system.a libboost_thread.a libglog.a libgflags.a libtcmalloc.a\
libconfig++.a libsnappy.a libyaml-cpp.a libleveldb.a libzmq.a)

BOSEN_LIBS := $(addprefix $(BOSEN_ROOT)/lib/, libpetuum-ml.a libpetuum-ps.a)

BOOST_LIBS := $(addprefix $(BOSEN_THIRD_PARTY_ROOT)/lib/, \
libboost_program_options.a)

# LIBS :=

# Import bosen settings
PETUUM_ROOT := $(BOSEN_ROOT)
-include $(PETUUM_ROOT)/defns.mk
