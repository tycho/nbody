top-level-make:

uname_S := $(shell uname -s 2>/dev/null || echo "not")
uname_M := $(shell uname -m 2>/dev/null || echo "not")
uname_O := $(shell uname -o 2>/dev/null || echo "not")

ifeq ($(uname_S),FreeBSD)
SHELL      := /usr/local/bin/bash
else
SHELL      := /bin/bash
endif
MAKEFLAGS  += --no-print-directory -Rr
.SUFFIXES:

ifneq ($(findstring $(MAKEFLAGS),s),s)
ifndef V
    QUIET_AR        = @echo '   ' AR   $@;
    QUIET_CC        = @echo '   ' CC   $@;
    QUIET_CXX       = @echo '   ' CXX  $@;
    QUIET_NVCC      = @echo '   ' NVCC $@;
    QUIET_LINK      = @echo '   ' LD   $@;
    QUIET           = @
    export V
endif
endif

OSNAME     := $(shell uname -s || echo "not")
ifneq ($(findstring CYGWIN,$(OSNAME)),)
OSNAME     := Cygwin
endif

# cc-option: Check if compiler supports first option, else fall back to second.
#
# This is complicated by the fact that unrecognised -Wno-* options:
#   (a) are ignored unless the compilation emits a warning; and
#   (b) even then produce a warning rather than an error
# To handle this we do a test compile, passing the option-under-test, on a code
# fragment that will always produce a warning (integer assigned to pointer).
# We then grep for the option-under-test in the compiler's output, the presence
# of which would indicate an "unrecognized command-line option" warning/error.
#
# Usage: cflags-y += $(call cc-option,$(CC),-march=winchip-c6,-march=i586)
ifeq (,$(findstring clean,$(MAKECMDGOALS)))
cxx-option = $(shell if test -z "`echo 'char *p=(char *)1;' | \
              $(1) $(2) -S -o /dev/null -xc++ - 2>&1 | grep -- $(2) -`"; \
              then echo "$(2)"; else echo "$(3)"; fi ;)
else
cxx-option =
endif

CXX        := g++
LINK       := $(CXX)
AR         := ar
RM         := rm -f
CP         := cp

ifneq ($(findstring icpc,$(CXX)),)
AR         := xiar
endif

export AR

ifneq ($(findstring icpc,$(CXX)),)
CFOPTIMIZE ?= -Ofast -xHOST -no-prec-sqrt
else
CFWARN     := -Wall \
              $(call cxx-option,$(CXX),-Werror=implicit,) \
              -Wmissing-declarations \
              -Wno-long-long \
              $(call cxx-option,$(CXX),-Wno-overlength-strings,) \
              -Wno-unknown-pragmas

ifndef DEBUG

# Good optimization flags for x86/x86_64
CFOPTIMIZE  = -O3 -march=native -ffast-math

ifneq ($(findstring armv7,$(uname_M)),)
# Good optimizations for modern ARMv7-a
CFOPTIMIZE  = -O3 -mcpu=native -mfpu=neon -mfloat-abi=hard -ffast-math
endif

else

CFOPTIMIZE  = -O0 -ggdb

endif # DEBUG

endif # icc
export CFOPTIMIZE

CPPFLAGS   += -D_GNU_SOURCE

CSTD       := $(call cxx-option,$(CXX),-std=c++11,-std=c++0x)

CFEXTRA    := -fno-strict-aliasing \
              -fvisibility=hidden -fvisibility-inlines-hidden \
              -fno-exceptions \
              -fno-rtti

CXXFLAGS   += $(CFOPTIMIZE) $(CSTD) $(CFEXTRA) $(CPPFLAGS) $(CFWARN)

ifeq ($(uname_S),Darwin)
ifneq ($(findstring gcc,$(shell $(CXX) -v 2>&1)),)
CXXFLAGS     += -Wa,-q
endif
endif

ifeq ($(uname_O),Cygwin)
CXXFLAGS += -fno-asynchronous-unwind-tables
LDFLAGS += -fno-asynchronous-unwind-tables
else
CXXFLAGS += -pthread
LDFLAGS += -pthread
endif
