BIN := nbody

all: $(BIN)

include ../common.mk

ifeq ($(shell type -P nvcc),)
NO_CUDA := 1
endif

ifdef HIGH_ENTROPY
CPPFLAGS  += -DHIGH_ENTROPY
endif

CPPFLAGS  += -DLIBTIME_STATIC

ifndef DEBUG
NVCCFLAGS := -O3 -Drestrict=
else
NVCCFLAGS := -O0 -g -Drestrict=
CPPFLAGS  += -DDEBUG
endif

ifneq ($(TRAVIS),)
ifneq ($(findstring clang,$(CC)),)
# Until the LLVM OpenMP runtime can be reliably found and used on Travis-CI.
OPENMP_FLAG :=
OPENMP_FLAG_FOUND := Yes
COMPILER_SUPPORTS_OPENMP :=
endif
endif

ifeq ($(NO_OPENMP),)

ifeq (,$(OPENMP_FLAG_FOUND))
OPENMP_FLAG := -fopenmp
OPENMP_FLAG_FOUND := \
	$(shell $(CC) $(OPENMP_FLAG) -E -dM -xc /dev/null 2>/dev/null | grep -q _OPENMP && echo "Yes")
endif

ifeq (,$(OPENMP_FLAG_FOUND))
OPENMP_FLAG := -fopenmp=libomp
OPENMP_FLAG_FOUND := \
	$(shell $(CC) $(OPENMP_FLAG) -E -dM -xc /dev/null 2>/dev/null | grep -q _OPENMP && echo "Yes")
endif

ifeq (,$(OPENMP_FLAG_FOUND))
OPENMP_FLAG := -fopenmp=libgomp
OPENMP_FLAG_FOUND := \
	$(shell $(CC) $(OPENMP_FLAG) -E -dM -xc /dev/null 2>/dev/null | grep -q _OPENMP && echo "Yes")
endif

ifneq (,$(OPENMP_FLAG_FOUND))
COMPILER_SUPPORTS_OPENMP := \
	$(shell $(CC) $(OPENMP_FLAG) -Werror -c -o /dev/null _cc_test_openmp_modern.c && echo "Yes")
endif

ifneq ($(COMPILER_SUPPORTS_OPENMP),)

# Hack for clang failing to understand "omp declare simd" in /usr/include/bits/math-vector.h
COMPILER_NEEDS_OPENMP_HACK := \
	$(shell $(CC) $(OPENMP_FLAG) -O3 -ffast-math -include math.h -c -xc /dev/null -o /dev/null 2>/dev/null || echo "Yes")
ifneq (,$(COMPILER_NEEDS_OPENMP_HACK))
$(warning Forcing OpenMP v3.1 to workaround compile failure in math.h)
# Force version to OpenMP 3.1
OPENMP_FLAG += -D_OPENMP=201107 -Wno-macro-redefined
endif

CPPFLAGS += -DUSE_OPENMP
CFLAGS += $(OPENMP_FLAG)
LIBOMP_PATHS := -L/usr/lib
#LIBOMP_PATHS := $(shell ldconfig -N -v 2>/dev/null | awk '{ if (NF == 1) { sub(/:$$/, "", $$1); path=$$1; } else if (match($$1, /^libomp.so$$/)) { print "-L"path; } }' )
LIBOMP_FOUND := $(shell $(CC) -c $(OPENMP_FLAG) openmp-test.c && $(LINK) -o openmp-test openmp-test.o $(LIBOMP_PATHS) -lomp >&2 && echo Yes)
ifeq ($(LIBOMP_FOUND),Yes)
LDFLAGS += $(LIBOMP_PATHS) -lomp
else
$(warning The Intel OpenMP runtime could not be found or is broken, using GNU OpenMP runtime)
LDFLAGS += $(OPENMP_FLAG)
endif # LIBOMP_FOUND
endif # COMPILER_SUPPORTS_OPENMP
endif # NO_OPENMP

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

CUDA_ROOT := $(dir $(shell which nvcc 2>/dev/null))..

LDFLAGS += -lm
ifeq ($(uname_S),Linux)
LDFLAGS += -lrt
endif

ifeq ($(CC),icc)
# Statically link Intel libraries so the executables are more portable
LDFLAGS += -static-intel
endif

ifneq ($(NO_CUDA),)
CPPFLAGS += -DNO_CUDA
else
GXX5_PATH := $(shell which g++-5 2>/dev/null)
NVCC := nvcc
ifneq ($(GXX5_PATH),)
NVCCFLAGS += -ccbin=$(GXX5_PATH)
endif
LDFLAGS += -L$(CUDA_ROOT)/lib -L$(CUDA_ROOT)/lib64 -lcudart
endif

ifndef NATIVE_C11
CPPFLAGS += -DUSE_LIBC11
LDFLAGS += ../libc11/libc11.a
endif
LDFLAGS += ../libtime/libtime.a

ifneq (,$(findstring CYGWIN,$(uname_S)))
LDFLAGS += -lwinmm
endif
ifneq (,$(findstring MINGW,$(uname_S)))
LDFLAGS += -lwinmm
endif

ifneq ($(findstring nvcc,$(NVCC)),)
ifeq ($(TRAVIS),)
# There's a list of compute capabilities and the corresponding compatible GPUs
# here: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
#
# I've chosen to enable three of them, with a relatively broad compatibility set
# for hardware made in the past few years.
#
NVCCFLAGS += \
	-gencode=arch=compute_52,code=\"sm_52,compute_52\" \
	-gencode=arch=compute_61,code=\"sm_61,compute_61\" \
	-gencode=arch=compute_70,code=\"sm_70,compute_70\"
else
# Only use one target architecture for Travis builds. We just want to make sure
# it compiles, not runs.
NVCCFLAGS += \
    -gencode=arch=compute_30,code=\"sm_30,compute_30\"
endif
endif

ifndef NATIVE_C11
CPPFLAGS += -I../libc11/include
endif
CPPFLAGS += -I../libtime/include

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
	nbody_util.c

ifeq ($(NO_CUDA),)
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

VALID_DEPS := $(shell ls $(DEPS) 2>/dev/null)
-include $(VALID_DEPS)

info: info-flags info-compiler info-os info-cpu

info-flags:
	$(QUIET)echo "CC      = $(CC)"
	$(QUIET)echo "LINK    = $(LINK)"
	$(QUIET)echo "CFLAGS  = $(CFLAGS)" | fmt -t -w 76
	$(QUIET)echo "LDFLAGS = $(LDFLAGS)" | fmt -t -w 76
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

.PHONY: clean info info-flags info-compiler info-os info-cpu

ifndef NO_CUDA
%.o: %.cu .cflags
	$(QUIET_NVCC)$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c -o $@ $<
else
%.o: %.cu .cflags
	$(QUIET_CC)$(CC) $(CFLAGS) $(INLINE_DEPGEN_CFLAGS) -xc -c -o $@ $<
endif

%.o: %.c .cflags
	$(QUIET_CC)$(CC) $(CFLAGS) $(INLINE_DEPGEN_CFLAGS) -c -o $@ $<

$(BIN): $(OBJECTS) .cflags
	$(QUIET_LINK)$(LINK) -o $@ $(OBJECTS) $(LDFLAGS) $(CFLAGS)

ifeq (,$(findstring clean,$(MAKECMDGOALS)))

TRACK_CFLAGS = $(subst ','\'',$(CC) $(CXX) $(NVCC) $(LINK) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(NVCCFLAGS) $(LDFLAGS))

.cflags: .force-cflags
	@FLAGS='$(TRACK_CFLAGS)'; \
	if test x"$$FLAGS" != x"`cat .cflags 2>/dev/null`" ; then \
		echo "    * rebuilding $(BIN): new build flags or prefix"; \
		echo "$$FLAGS" > .cflags; \
	fi

.PHONY: .force-cflags

endif
