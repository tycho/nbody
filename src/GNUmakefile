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

CFLAGS    += -fPIC
LDFLAGS   += -fPIC

ifneq ($(findstring clang,$(CC)),)
  GCC := gcc
  GCC_MACHINE := $(shell $(GCC) -dumpmachine)
  GCC_VERSION := $(shell $(GCC) -dumpversion)
  GCC_HEADER_PATH := /usr/lib/gcc/$(GCC_MACHINE)/$(GCC_VERSION)/include
  ifneq ($(GCC_HEADER_PATH),)
    CLANG_GCC_HEADER_FLAGS := -idirafter "$(GCC_HEADER_PATH)"
    CFLAGS += -idirafter "$(GCC_HEADER_PATH)"
  endif
endif

ifndef NO_OPENMP
ifeq (,$(findstring clean,$(MAKECMDGOALS)))
openmp.mk: .cc
	$(QUIET)./openmp-detect.sh "$(CC) $(CLANG_GCC_HEADER_FLAGS)"
-include openmp.mk
endif

ifneq ($(OPENMP_SUPPORTED),)
  ifeq ($(OPENMP_SUPPORTED),Yes)
    CPPFLAGS += -DUSE_OPENMP
    CFLAGS   += $(OPENMP_CFLAGS)
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
    CFLAGS += -maltivec
  endif
  ifeq ($(uname_M),armv7l)
    CPPFLAGS += -DHAVE_SIMD
    CFLAGS += -marm -mfpu=neon
  endif
  ifeq ($(uname_M),x86_64)
    CPPFLAGS += -DHAVE_SIMD
  endif
  ifeq ($(uname_M),i686)
    CPPFLAGS += -DHAVE_SIMD
  endif
endif

ifeq ($(CC),icc)
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
CFLAGS        += $(SDL_CFLAGS) $(GLEW_CFLAGS)
LDFLAGS       += $(SDL_LDFLAGS) $(GLEW_LDFLAGS)
ifneq (,$(findstring MINGW,$(uname_S)))
LDFLAGS       += -lglu32 -lopengl32 -lmingw32
endif
endif

ifndef NATIVE_C11
  CPPFLAGS += -DUSE_LIBC11
  LDFLAGS += ../subprojects/c11/libc11.a
endif
LDFLAGS += ../subprojects/time/libtime.a

LDFLAGS += -lm
ifeq ($(uname_S),Linux)
  LDFLAGS += -lrt
endif

ifneq (,$(findstring CYGWIN,$(uname_S)))
  LDFLAGS += -lwinmm
endif
ifneq (,$(findstring MINGW,$(uname_S)))
  CFLAGS += -fno-asynchronous-unwind-tables
  LDFLAGS += -mconsole -lwinmm
endif

ifdef CUDA
  ifndef DEBUG
    NVCCFLAGS := -O3
  else
    NVCCFLAGS := -O0 -g
  endif
  ifneq (,$(findstring clang,$(NVCC)))
    NVCCFLAGS += -Drestrict=__restrict
    NVCCFLAGS += -fcuda-flush-denormals-to-zero
    NVCCFLAGS += --cuda-gpu-arch=sm_52 \
                 --cuda-gpu-arch=sm_61 \
                 --cuda-gpu-arch=sm_70 \
                 --cuda-gpu-arch=sm_75
  else
    NVCCFLAGS += -Drestrict=
    NVCCFLAGS += --ftz true
    NVCCFLAGS += -Xcompiler -fPIC

    NVCCTARGETS := $(shell ./nvcc-flags.sh $(NVCC))
    ifeq ($(NVCCTARGETS),)
        $(error nvcc not found or too old, aborting)
    endif
    NVCCFLAGS += $(NVCCTARGETS)
  endif
endif

ifndef NATIVE_C11
  CPPFLAGS += -I../subprojects/c11/include
endif
CPPFLAGS += -I../subprojects/time/include

SOURCES := \
	nbody.cu \
	nbody_CPU_AOS.c \
	nbody_CPU_AOS_tiled.c \
	nbody_CPU_SOA.c \
	nbody_CPU_SOA_tiled.c \
	nbody_CPU_AltiVec.c \
	nbody_CPU_NEON.c \
	nbody_CPU_SSE.c \
	nbody_CPU_AVX.c \
	nbody_util.c \
	nbody_render_gl.c

ifdef CUDA
SOURCES += \
	nbody_GPU_shared.cu \
	nbody_multiGPU.cu
endif

OBJECTS := $(SOURCES:%.cu=%.o)
OBJECTS := $(OBJECTS:%.c=%.o)

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

info: info-flags info-compiler info-os info-cpu

info-flags:
	$(QUIET)echo "CC        = $(CC)"
	$(QUIET)echo "NVCC      = $(NVCC)"
	$(QUIET)echo "CUDA      = $(CUDA)"
	$(QUIET)echo "LINK      = $(LINK)"
	$(QUIET)echo "CFLAGS    = $(CFLAGS)" | fmt -t -w 76
	$(QUIET)echo "NVCCFLAGS = $(NVCCFLAGS)" | fmt -t -w 76
	$(QUIET)echo "LDFLAGS   = $(LDFLAGS)" | fmt -t -w 76
	$(QUIET)echo

info-compiler:
	$(QUIET)echo "Compiler version: $(shell $(CC) --version 2>&1 | head -n 1)"
ifneq ($(VERBOSE),)
	$(QUIET)echo "Compiler invocation details:"
	$(QUIET)gcc -### -o /dev/null -xc /dev/null 2>&1 | fmt -t -w 72
endif
	$(QUIET)echo

info-os:
	$(QUIET)echo "OS version: $(shell uname -m -s -r)"

info-cpu:
	$(QUIET)echo "CPU information:"
	$(QUIET)lscpu | fmt -t -w 72 | sed 's/^/    /'
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
	$(QUIET_CC)$(CC) $(CFLAGS) $(INLINE_DEPGEN_CFLAGS) -xc -c -o $@ $<
endif

%.o: %.c $(REBUILD_SENTINELS)
	$(QUIET_CC)$(CC) $(CFLAGS) $(INLINE_DEPGEN_CFLAGS) -c -o $@ $<

$(BIN): $(OBJECTS) $(REBUILD_SENTINELS)
	$(QUIET_LINK)$(LINK) -o $@ $(OBJECTS) $(LDFLAGS) $(CFLAGS)

ifeq (,$(findstring clean,$(MAKECMDGOALS)))

TRACK_CCONLY = $(subst ','\'',$(CC) $(CXX) $(NVCC) $(LINK))
TRACK_CFLAGS = $(subst ','\'',$(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NVCCFLAGS) $(LDFLAGS))

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
