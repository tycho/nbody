BIN := nbody

all: $(BIN)

SHELL := /bin/bash

uname_S := $(shell uname -s 2>/dev/null || echo "not")
uname_M := $(shell uname -m 2>/dev/null || echo "not")
uname_O := $(shell uname -o 2>/dev/null || echo "not")

ifeq ($(shell type -P nvcc),)
NO_CUDA := 1
endif

CC := gcc
CXX := g++
NVCC := nvcc

DEBUG = 0

NVCCFLAGS := -O3
CFOPTIMIZE := -O3 -ffast-math

ifneq ($(DEBUG),0)
CFOPTIMIZE := -O0 -ggdb
NVCCFLAGS := -O0 -g
endif

CPPFLAGS := -D_GNU_SOURCE
CFLAGS   := $(CFOPTIMIZE) -std=c11 -Wall -Wdeclaration-after-statement -Wimplicit-function-declaration -Wmissing-declarations -Wmissing-prototypes -Wno-unknown-pragmas -Wno-long-long -Wno-overlength-strings -Wno-declaration-after-statement -Wold-style-definition -Wstrict-prototypes
CXXFLAGS := $(CFOPTIMIZE) -fno-exceptions -fno-rtti -fno-strict-aliasing -Wall -Wunused -Wmissing-declarations -Wno-unknown-pragmas -Wno-long-long -Wno-overlength-strings

ifeq ($(uname_S),Darwin)
ifneq ($(findstring gcc,$(shell $(CXX) -v 2>&1)),)
CXXFLAGS += -Wa,-q
endif
endif

ifeq ($(NO_OPENMP),)
COMPILER_SUPPORTS_OPENMP := $(shell $(CXX) -fopenmp -E -dM -xc /dev/null 2>/dev/null | grep _OPENMP)
ifneq ($(COMPILER_SUPPORTS_OPENMP),)
CPPFLAGS += -DUSE_OPENMP
CFLAGS += -fopenmp
CXXFLAGS += -fopenmp
LDFLAGS += -fopenmp
LIBIOMP5_FOUND := $(shell gcc -o /dev/null -xc <(echo "int main() { }") -liomp5 2>/dev/null && echo Yes)
ifeq ($(LIBIOMP5_FOUND),Yes)
LDFLAGS += -liomp5
else
LIBIOMP5_PATH := $(shell ldconfig -N -v 2>/dev/null | awk '{ if (NF == 1) { sub(/:$$/, "", $$1); path=$$1; } else if (match($$1, /^libiomp5.so$$/)) { print path; } }' )
ifneq ($(LIBIOMP5_PATH),)
LDFLAGS += -L$(LIBIOMP5_PATH) -liomp5
endif # LIBIOMP5_PATH
endif # LIBIOMP5_FOUND
endif # COMPILER_SUPPORTS_OPENMP
endif # NO_OPENMP

ifeq ($(NO_SIMD),)
ifeq ($(uname_M),ppc)
CPPFLAGS += -DHAVE_ALTIVEC
CFLAGS += -maltivec
CXXFLAGS += -maltivec
endif
ifeq ($(uname_M),armv7l)
CPPFLAGS += -DHAVE_NEON
CFLAGS += -marm -mfpu=neon
CXXFLAGS += -marm -mfpu=neon
endif
ifeq ($(uname_M),x86_64)
CPPFLAGS += -DHAVE_SSE
endif
endif

ifneq ($(DEBUG),0)
CPPFLAGS += -DDEBUG=1
endif

CUDA_ROOT := $(dir $(shell which nvcc 2>/dev/null))..

ifneq ($(uname_O),Cygwin)
CXXFLAGS += -pthread
CFLAGS += -pthread
LDFLAGS += -pthread
endif

ifeq ($(uname_S),Linux)
LDFLAGS += -lrt
endif

ifeq ($(CXX),icpc)
# Special Intel compiler options to give us more detail
CXXFLAGS += -parallel -openmp-report -vec-report
CFLAGS += -parallel -openmp-report -vec-report

# Statically link Intel libraries so the executables are more portable
LDFLAGS += -static-intel
endif

ifneq ($(NO_CUDA),)
NVCC := $(CC)
NVCCFLAGS := $(CFLAGS)
NVCCFLAGS += -xc
CPPFLAGS += -DNO_CUDA
else
NVCCFLAGS +=
LDFLAGS += -L$(CUDA_ROOT)/lib -L$(CUDA_ROOT)/lib64 -lcudart
endif

LDFLAGS += -Llibc11 -lc11 -Llibtime -ltime

ifneq ($(findstring nvcc,$(NVCC)),)
NVCCFLAGS += -gencode=arch=compute_10,code=\"sm_10,compute_10\" -gencode=arch=compute_20,code=\"sm_20,compute_20\" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\"
endif

LINK := $(CXX)

INCLUDE := -Ilibc11/include -Ilibtime

SOURCES := \
	src/nbody.cu \
	src/nbody_CPU_AOS.c \
	src/nbody_CPU_AOS_tiled.c \
	src/nbody_CPU_SOA.c \
	src/nbody_CPU_SOA_tiled.c \
	src/nbody_CPU_AltiVec.c \
	src/nbody_CPU_NEON.c \
	src/nbody_CPU_SSE.c

ifeq ($(NO_CUDA),)
SOURCES += \
	src/nbody_GPU_shared.cu \
	src/nbody_multiGPU.cu
endif

OBJECTS := $(SOURCES:%.cu=%.o)
OBJECTS := $(OBJECTS:%.cpp=%.o)
OBJECTS := $(OBJECTS:%.c=%.o)

ifneq ($(findstring $(MAKEFLAGS),s),s)
ifndef V
        QUIET_CC        = @echo '   ' CC   $@;
        QUIET_CXX       = @echo '   ' CXX  $@;
        QUIET_NVCC      = @echo '   ' NVCC $@;
        QUIET_LINK      = @echo '   ' LD   $@;
        QUIET           = @
        export V
endif
endif

clean:
	rm -f $(BIN)
	rm -f $(OBJECTS)

%.o: %.cu .cflags
	$(QUIET_NVCC)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(INCLUDE) -c -o $@ $<

%.o: %.cpp .cflags
	$(QUIET_CXX)$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDE) -c -o $@ $<

%.o: %.c .cflags
	$(QUIET_CC)$(CC) $(CPPFLAGS) $(CFLAGS) $(INCLUDE) -c -o $@ $<

$(BIN): $(OBJECTS) libc11/libc11.a libtime/libtime.a .cflags
	$(QUIET_LINK)$(LINK) -o $@ $(OBJECTS) $(LDFLAGS)

libtime/libtime.a:
	$(MAKE) -C libtime libtime.a

libc11/libc11.a:
	$(MAKE) -C libc11 libc11.a

ifeq (,$(findstring clean,$(MAKECMDGOALS)))

TRACK_CFLAGS = $(subst ','\'',$(CC) $(CXX) $(NVCC) $(LINK) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NVCCFLAGS) $(INCLUDE) $(LDFLAGS))

.cflags: .force-cflags
	@FLAGS='$(TRACK_CFLAGS)'; \
	if test x"$$FLAGS" != x"`cat .cflags 2>/dev/null`" ; then \
		echo "    * rebuilding $(BIN): new build flags or prefix"; \
		echo "$$FLAGS" > .cflags; \
	fi

.PHONY: .force-cflags

endif
