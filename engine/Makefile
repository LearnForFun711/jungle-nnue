# Makefile for the Jungle NNUE engine.
#
#   make            build everything into build/
#   make clean      remove build artefacts
#   make test       build and run the rule cross-check
#
# The build is plain C++17 with no external dependencies.  -O3 is important:
# the MCTS teacher and the search are both very hot loops.

CXX      ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter -funroll-loops -fno-math-errno
LDFLAGS  ?=
BUILD    := build

# Native tuning helps a lot on the machines we actually run on.  It is optional:
# override NATIVE=0 for a portable binary.
ifeq ($(NATIVE),1)
  CXXFLAGS += -march=native
endif

TOOLS := selfplay match dump_moves bench eval_nnue nnue_check teacher_probe nnue_ref
LIB   := libjungle

# The GPU teacher.  These are the same code the CUDA extension compiles, but
# built for the CPU: the portable core has to be testable on a machine
# without a GPU, and it is the code the GPU runs.
GPU_TOOLS := gpu_parity gpu_selfplay_cpu gpu_teacher_probe
GPU_SRCS  := gpu/gpu_parity.cpp gpu/gpu_selfplay_cpu.cpp gpu/gpu_teacher_probe.cpp \
             gpu/jungle_gpu_common.cpp

ifeq ($(shell uname),Darwin)
  LIBFILE := $(BUILD)/$(LIB).dylib
  LIBFLAGS := -dynamiclib
else
  LIBFILE := $(BUILD)/$(LIB).so
  LIBFLAGS := -shared
endif

# A shared object needs position-independent code.  macOS ignores -fPIC, so
# it is harmless there, but Linux refuses to link the library without it.
LIBFLAGS += -fPIC

all: $(addprefix $(BUILD)/,$(TOOLS)) $(LIBFILE)

$(BUILD)/%: tools/%.cpp $(wildcard src/*.hpp)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) -lpthread

$(LIBFILE): tools/jungle_api.cpp $(wildcard src/*.hpp)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(LIBFLAGS) -o $@ $< $(LDFLAGS) -lpthread

test: $(BUILD)/dump_moves
	@python3 ../scripts/verify_rules.py

# The GPU teacher's CPU build.  `make gpu` builds the three tools that check
# the portable core against the engine and that generate data without a GPU.
gpu: $(addprefix $(BUILD)/,$(GPU_TOOLS))

$(BUILD)/gpu_parity: gpu/gpu_parity.cpp gpu/jungle_gpu_common.cpp $(wildcard gpu/*.cuh) $(wildcard gpu/*.h) $(wildcard src/*.hpp)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ gpu/gpu_parity.cpp gpu/jungle_gpu_common.cpp $(LDFLAGS)

$(BUILD)/gpu_selfplay_cpu: gpu/gpu_selfplay_cpu.cpp gpu/jungle_gpu_common.cpp $(wildcard gpu/*.cuh) $(wildcard gpu/*.h) $(wildcard src/*.hpp)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ gpu/gpu_selfplay_cpu.cpp gpu/jungle_gpu_common.cpp $(LDFLAGS)

$(BUILD)/gpu_teacher_probe: gpu/gpu_teacher_probe.cpp gpu/jungle_gpu_common.cpp $(wildcard gpu/*.cuh) $(wildcard gpu/*.h) $(wildcard src/*.hpp)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ gpu/gpu_teacher_probe.cpp gpu/jungle_gpu_common.cpp $(LDFLAGS)

clean:
	rm -rf $(BUILD)

.PHONY: all clean test gpu
