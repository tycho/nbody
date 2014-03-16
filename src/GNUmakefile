BIN := nbody

all: $(BIN)

include ../common.mk

ifeq ($(shell type -P nvcc),)
NO_CUDA := 1
endif

ifndef DEBUG
NVCCFLAGS := -O3
else
NVCCFLAGS := -O0 -g
CPPFLAGS  += -DDEBUG
endif

ifeq ($(NO_OPENMP),)
COMPILER_SUPPORTS_OPENMP := $(shell $(CC) -fopenmp -E -dM -xc /dev/null 2>/dev/null | grep _OPENMP)
ifneq ($(COMPILER_SUPPORTS_OPENMP),)
CPPFLAGS += -DUSE_OPENMP
CFLAGS += -fopenmp
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
endif
ifeq ($(uname_M),armv7l)
CPPFLAGS += -DHAVE_NEON
CFLAGS += -marm -mfpu=neon
endif
ifeq ($(uname_M),x86_64)
CPPFLAGS += -DHAVE_SSE
endif
endif

CUDA_ROOT := $(dir $(shell which nvcc 2>/dev/null))..

LDFLAGS += -lm
ifeq ($(uname_S),Linux)
LDFLAGS += -lrt
endif

ifeq ($(CC),icc)
# Special Intel compiler options to give us more detail
CFLAGS += -parallel -openmp-report -vec-report

# Statically link Intel libraries so the executables are more portable
LDFLAGS += -static-intel
endif

ifneq ($(NO_CUDA),)
QUIET_NVCC = $(QUIET_CC)
NVCC := $(CC)
NVCCFLAGS := $(CFLAGS)
NVCCFLAGS += -xc
CPPFLAGS += -DNO_CUDA
else
NVCC := nvcc
NVCCFLAGS +=
LDFLAGS += -L$(CUDA_ROOT)/lib -L$(CUDA_ROOT)/lib64 -lcudart
endif

LDFLAGS += -L../libc11 -lc11 -L../libtime -ltime

ifneq ($(findstring nvcc,$(NVCC)),)
NVCCFLAGS += -gencode=arch=compute_10,code=\"sm_10,compute_10\" -gencode=arch=compute_20,code=\"sm_20,compute_20\" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\"
endif

INCLUDE := -I../libc11/include -I../libtime/include

SOURCES := \
	nbody.cu \
	nbody_CPU_AOS.c \
	nbody_CPU_AOS_tiled.c \
	nbody_CPU_SOA.c \
	nbody_CPU_SOA_tiled.c \
	nbody_CPU_AltiVec.c \
	nbody_CPU_NEON.c \
	nbody_CPU_SSE.c

ifeq ($(NO_CUDA),)
SOURCES += \
	nbody_GPU_shared.cu \
	nbody_multiGPU.cu
endif

OBJECTS := $(SOURCES:%.cu=%.o)
OBJECTS := $(OBJECTS:%.c=%.o)

clean:
	rm -f $(BIN)
	rm -f $(OBJECTS)

%.o: %.cu .cflags
	$(QUIET_NVCC)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(INCLUDE) -c -o $@ $<

%.o: %.c .cflags
	$(QUIET_CC)$(CC) $(CPPFLAGS) $(CFLAGS) $(INCLUDE) -c -o $@ $<

$(BIN): $(OBJECTS) .cflags
	$(QUIET_LINK)$(LINK) -o $@ $(OBJECTS) $(LDFLAGS)

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
