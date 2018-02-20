include common.mk

BIN := nbody

ifndef NATIVE_C11
DEPS += subprojects/c11/libc11.a
endif
DEPS += subprojects/time/libtime.a

ifeq ($(strip $(MAKE_JOBS)),)
    ifeq ($(uname_S),Darwin)
        CPUS := $(shell /usr/sbin/sysctl -n hw.ncpu)
    endif
    ifeq ($(uname_S),Linux)
        CPUS := $(shell grep ^processor /proc/cpuinfo | wc -l)
    endif
    ifneq (,$(findstring MINGW,$(uname_S))$(findstring CYGWIN,$(uname_S)))
        CPUS := $(shell getconf _NPROCESSORS_ONLN)
    endif
    MAKE_JOBS := $(CPUS)
endif

ifeq ($(strip $(MAKE_JOBS)),)
    MAKE_JOBS := 8
endif

top-level-make:
	@$(MAKE) -f GNUmakefile -j$(MAKE_JOBS) all

all: $(BIN)

info:
	$(QUIET)$(MAKE) -C src $@

info%:
	$(QUIET)$(MAKE) -C src $@

$(BIN): src/$(BIN)
	$(QUIET)$(CP) $< $@

distclean: clean

clean:
	$(RM) $(BIN)
	$(MAKE) -C src clean
	$(MAKE) -C subprojects/time clean
	$(MAKE) -C subprojects/c11 clean

src/$(BIN): $(DEPS)
	$(QUIET)$(MAKE) -C src $(BIN)
.PHONY: src/$(BIN)

subprojects/time/.git:
	git submodule update --init subprojects/time

subprojects/time/libtime.a: subprojects/time/.git
	$(QUIET)$(MAKE) -C subprojects/time libtime.a
.PHONY: subprojects/time/libtime.a

ifndef NATIVE_C11
subprojects/c11/.git:
	git submodule update --init subprojects/c11

subprojects/c11/libc11.a: subprojects/c11/.git
	$(QUIET)$(MAKE) -C subprojects/c11 libc11.a
.PHONY: subprojects/c11/libc11.a
endif

