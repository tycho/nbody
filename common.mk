all:

SHELL      := /bin/bash
MAKEFLAGS  += --no-print-directory -Rr
.SUFFIXES:

uname_S := $(shell uname -s 2>/dev/null || echo "not")
uname_M := $(shell uname -m 2>/dev/null || echo "not")
uname_O := $(shell uname -o 2>/dev/null || echo "not")

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

ifneq ($(shell type -P clang),)
CC         := clang
else
CC         := gcc
endif

LINK       := $(CC)
AR         := ar rcu
RM         := rm -f
CP         := cp

CFWARN     := \
	-Wall \
	-Wdeclaration-after-statement \
	-Wimplicit-function-declaration \
	-Wmissing-declarations \
	-Wmissing-prototypes \
	-Wno-declaration-after-statement \
	-Wno-long-long \
	-Wno-overlength-strings \
	-Wno-unknown-pragmas \
	-Wold-style-definition \
	-Wstrict-prototypes
CPPFLAGS   := -D_GNU_SOURCE $(CFWARN)
CFOPTIMIZE := -O3 -ffast-math
CFLAGS     := $(CFOPTIMIZE) -std=gnu11 -fno-strict-aliasing $(CPPFLAGS)

ifeq ($(uname_S),Darwin)
ifneq ($(findstring gcc,$(shell $(CC) -v 2>&1)),)
CFLAGS     += -Wa,-q
endif
endif

ifneq ($(uname_O),Cygwin)
CFLAGS += -pthread
LDFLAGS += -pthread
endif
