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
cc-option = $(shell if test -z "`echo 'void*p=1;' | \
              $(1) $(2) -S -o /dev/null -xc - 2>&1 | grep -- $(2) -`"; \
              then echo "$(2)"; else echo "$(3)"; fi ;)
else
cc-option =
endif

# cc-option-add: Add an option to compilation flags, but only if supported.
# Usage: $(call cc-option-add,CFLAGS,CC,-march=winchip-c6)
cc-option-add = $(eval $(call cc-option-add-closure,$(1),$(2),$(3)))
define cc-option-add-closure
    ifneq ($$(call cc-option,$$($(2)),$(3),n),n)
        $(1) += $(3)
    endif
endef

ifneq ($(shell type -P gcc5),)
# For FreeBSD
CC         := gcc5
else
CC         := gcc
endif

LINK       := $(CC)
AR         := ar
RM         := rm -f
CP         := cp

ifneq ($(findstring icc,$(CC)),)
AR         := xiar
endif

export AR

ifneq ($(findstring icc,$(CC)),)
CFOPTIMIZE ?= -Ofast -xHOST -no-prec-sqrt
else
CFWARN     := -Wall \
              -Wdeclaration-after-statement \
              $(call cc-option,$(CC),-Werror=implicit,) \
              $(call cc-option,$(CC),-Werror=undef,) \
              -Wmissing-declarations \
              -Wmissing-prototypes \
              -Wno-declaration-after-statement \
              -Wno-long-long \
              $(call cc-option,$(CC),-Wno-overlength-strings,) \
              -Wno-unknown-pragmas \
              -Wold-style-definition \
              -Wstrict-prototypes

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

CSTD       := $(call cc-option,$(CC),-std=gnu11,-std=gnu99)

CFEXTRA    := -fno-strict-aliasing

CFLAGS     += $(CFOPTIMIZE) $(CSTD) $(CFEXTRA) $(CPPFLAGS) $(CFWARN)

ifeq ($(uname_S),Darwin)
ifneq ($(findstring gcc,$(shell $(CC) -v 2>&1)),)
CFLAGS     += -Wa,-q
endif
endif

ifeq ($(uname_O),Cygwin)
CFLAGS += -fno-asynchronous-unwind-tables
LDFLAGS += -fno-asynchronous-unwind-tables
else
CFLAGS += -pthread
LDFLAGS += -pthread
endif
