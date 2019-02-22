BIN := nbody

all: $(BIN)

include common.mk

ifdef HIGH_ENTROPY
CPPFLAGS  += -DHIGH_ENTROPY
endif

CPPFLAGS  += -DLIBTIME_STATIC

ifdef DEBUG
CPPFLAGS  += -DDEBUG
endif

CXXFLAGS  += -fPIC
LDFLAGS   += -fPIC

ifneq ($(findstring clang,$(CXX)),)
  GCC := gcc
  GCC_MACHINE := $(shell $(GCC) -dumpmachine)
  GCC_VERSION := $(shell $(GCC) -dumpversion)
  GCC_HEADER_PATH := /usr/lib/gcc/$(GCC_MACHINE)/$(GCC_VERSION)/include
  ifneq ($(GCC_HEADER_PATH),)
    CLANG_GCC_HEADER_FLAGS := -idirafter "$(GCC_HEADER_PATH)"
    CXXFLAGS += -idirafter "$(GCC_HEADER_PATH)"
  endif
endif

ifndef NO_OPENMP
ifeq (,$(findstring info,$(MAKECMDGOALS)))
ifeq (,$(findstring clean,$(MAKECMDGOALS)))
openmp.mk: .cc openmp-detect.sh
	$(QUIET)./openmp-detect.sh "$(CXX) $(CLANG_GCC_HEADER_FLAGS)"
-include openmp.mk
endif
endif

ifneq ($(OPENMP_SUPPORTED),)
  ifeq ($(OPENMP_SUPPORTED),Yes)
    CPPFLAGS += -DUSE_OPENMP
    CXXFLAGS += $(OPENMP_CFLAGS)
    LDFLAGS  += $(OPENMP_LIBS)
  else
    ifeq (,$(findstring clean,$(MAKECMDGOALS)))
      $(warning Could not detect OpenMP support in your compiler. Resulting binary will be slow.)
    endif
  endif
endif
endif

ifeq ($(NO_SIMD),)
  ifeq ($(uname_M),ppc)
    CPPFLAGS += -DHAVE_SIMD
    CXXFLAGS += -maltivec
  endif
  ifeq ($(uname_M),armv7l)
    CPPFLAGS += -DHAVE_SIMD
    CXXFLAGS += -marm -mfpu=neon
  endif
  ifeq ($(uname_M),x86_64)
    CPPFLAGS += -DHAVE_SIMD
  endif
  ifeq ($(uname_M),i686)
    CPPFLAGS += -DHAVE_SIMD
  endif
endif

ifeq ($(CXX),icpc)
  # Statically link Intel libraries so the executables are more portable
  LDFLAGS += -static-intel
endif

ifdef CUDA
  CPPFLAGS  += -DUSE_CUDA
  NVCC      := nvcc
  CUDA_ROOT := $(dir $(shell which nvcc 2>/dev/null))..
  LDFLAGS   += -L$(CUDA_ROOT)/lib -L$(CUDA_ROOT)/lib64 -lcudart
endif

ifdef OPENGL
CPPFLAGS      += -DUSE_GL
SDL_CFLAGS    := $(shell pkg-config --cflags sdl2)
GLEW_CFLAGS   := $(shell pkg-config --cflags glew)
SDL_LDFLAGS   := $(shell pkg-config --libs sdl2)
GLEW_LDFLAGS  := $(shell pkg-config --libs glew)
CXXFLAGS      += $(SDL_CFLAGS) $(GLEW_CFLAGS)
LDFLAGS       += $(SDL_LDFLAGS) $(GLEW_LDFLAGS)
ifneq (,$(findstring MINGW,$(uname_S)))
LDFLAGS       += -lglu32 -lopengl32 -lmingw32
endif
endif

LDFLAGS += -lm
ifeq ($(uname_S),Linux)
  LDFLAGS += -lrt
endif

ifneq (,$(findstring CYGWIN,$(uname_S)))
  LDFLAGS += -lwinmm
endif
ifneq (,$(findstring MINGW,$(uname_S)))
  CXXFLAGS += -fno-asynchronous-unwind-tables
  LDFLAGS += -mconsole -lwinmm
endif

ifdef CUDA
  NVCCFLAGS += $(CSTD)
  ifneq (,$(findstring clang,$(NVCC)))
    NVCCFLAGS += $(CXXFLAGS)
    NVCCFLAGS += -fcuda-flush-denormals-to-zero
    NVCCFLAGS += --cuda-gpu-arch=sm_52 \
                 --cuda-gpu-arch=sm_61 \
                 --cuda-gpu-arch=sm_70 \
                 --cuda-gpu-arch=sm_75
  else
    NVCCFLAGS += -O3
    NVCCFLAGS += --ftz true
    NVCCFLAGS += -Xcompiler -fPIC

ifeq (,$(findstring info,$(MAKECMDGOALS)))
ifeq (,$(findstring clean,$(MAKECMDGOALS)))
nvcc-flags.mk: .cc
	$(QUIET)./nvcc-flags.sh $(NVCC)
-include nvcc-flags.mk
endif
endif
    ifdef NVCC_TARGET_FLAGS_DONE
    ifeq ($(NVCC_TARGET_FLAGS),)
        $(error nvcc not found or too old, aborting)
    endif
    endif
    NVCCFLAGS += $(NVCC_TARGET_FLAGS)
  endif
endif

SOURCES := \
	nbody.cu \
	nbody_CPU_AOS.cpp \
	nbody_CPU_AOS_tiled.cpp \
	nbody_CPU_SOA.cpp \
	nbody_CPU_SOA_tiled.cpp \
	nbody_CPU_AltiVec.cpp \
	nbody_CPU_NEON.cpp \
	nbody_CPU_SSE.cpp \
	nbody_CPU_AVX.cpp \
	nbody_util.cpp \
	nbody_render_gl.cpp

ifdef CUDA
SOURCES += \
	nbody_GPU_AOS.cu \
	nbody_GPU_AOS_const.cu \
	nbody_GPU_AOS_tiled.cu \
	nbody_GPU_AOS_tiled_const.cu \
	nbody_GPU_Atomic.cu \
	nbody_GPU_Shared.cu \
	nbody_GPU_Shuffle.cu \
	nbody_GPU_SOA_tiled.cu \
	nbody_multiGPU.cu
endif

OBJECTS := $(SOURCES:%.cu=%.o)
OBJECTS := $(OBJECTS:%.cpp=%.o)

# Dependency generation
#
ifndef NO_INLINE_DEPGEN
  # See info node: (gcc) Preprocessor Options
  INLINE_DEPGEN_CFLAGS = -MMD
endif

DEPS := $(OBJECTS:%.o=%.d)

REBUILD_SENTINELS := .cflags .cc

VALID_DEPS := $(shell ls $(DEPS) 2>/dev/null)
-include $(VALID_DEPS)

info: info-flags info-compiler info-os info-cpu info-gpu

info-flags:
	$(QUIET)echo "CC        = $(CC)"
	$(QUIET)echo "CXX       = $(CXX)"
	$(QUIET)echo "NVCC      = $(NVCC)"
	$(QUIET)echo "CUDA      = $(CUDA)"
	$(QUIET)echo "LINK      = $(LINK)"
	$(QUIET)echo "CXXFLAGS  = $(CXXFLAGS)" | fmt -t -w 76
	$(QUIET)echo "NVCCFLAGS = $(NVCCFLAGS)" | fmt -t -w 76
	$(QUIET)echo "LDFLAGS   = $(LDFLAGS)" | fmt -t -w 76
	$(QUIET)echo

info-compiler:
	$(QUIET)echo "Compiler version: $(shell $(CXX) --version 2>&1 | head -n 1)"
ifneq ($(VERBOSE),)
	$(QUIET)echo "Compiler invocation details:"
	$(QUIET)$(CXX) -### -o /dev/null -xc++ /dev/null 2>&1 | fmt -t -w 72
endif
	$(QUIET)echo

info-os:
	$(QUIET)echo "OS version: $(shell uname -m -s -r)"

info-cpu:
	$(QUIET)echo "CPU information:"
	$(QUIET)lscpu | fmt -t -w 72 | sed 's/^/    /'
	$(QUIET)echo

info-gpu:
	$(QUIET)echo "GPU information:"
	$(QUIET)nvidia-smi
	$(QUIET)echo

clean:
	rm -f $(BIN)
	rm -f $(OBJECTS)
	rm -f $(DEPS)
	rm -f .cflags .cc openmp.mk

.PHONY: clean info info-flags info-compiler info-os info-cpu

ifdef CUDA
%.o: %.cu $(REBUILD_SENTINELS)
	$(QUIET_NVCC)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $<
else
%.o: %.cu $(REBUILD_SENTINELS)
	$(QUIET_CXX)$(CXX) $(CXXFLAGS) $(INLINE_DEPGEN_CFLAGS) -xc++ -c -o $@ $<
endif

%.o: %.cpp $(REBUILD_SENTINELS)
	$(QUIET_CXX)$(CXX) $(CXXFLAGS) $(INLINE_DEPGEN_CFLAGS) -c -o $@ $<

$(BIN): $(OBJECTS) $(REBUILD_SENTINELS)
	$(QUIET_LINK)$(LINK) -o $@ $(OBJECTS) $(LDFLAGS) $(CXXFLAGS)

ifeq (,$(findstring info,$(MAKECMDGOALS)))
ifeq (,$(findstring clean,$(MAKECMDGOALS)))

TRACK_CCONLY = $(subst ','\'',$(CC) $(CXX) $(NVCC) $(LINK))
TRACK_CFLAGS = $(subst ','\'',$(CXXFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NVCCFLAGS) $(LDFLAGS))

.cc: .force-cflags
	@FLAGS='$(TRACK_CCONLY)'; \
	if test x"$$FLAGS" != x"`cat .cc 2>/dev/null`" ; then \
		echo "    * rebuilding $(BIN): new compiler"; \
		echo "$$FLAGS" > .cc; \
	fi

.cflags: .cc .force-cflags
	@FLAGS='$(TRACK_CFLAGS)'; \
	if test x"$$FLAGS" != x"`cat .cflags 2>/dev/null`" ; then \
		echo "    * rebuilding $(BIN): new build flags"; \
		echo "$$FLAGS" > .cflags; \
	fi

.PHONY: .force-cflags

endif
endif
