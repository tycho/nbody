include common.mk

BIN := nbody

ifndef NATIVE_C11
DEPS += libc11/libc11.a
endif
DEPS += libtime/libtime.a

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
	$(MAKE) -C libtime clean
	$(MAKE) -C libc11 clean

src/$(BIN): $(DEPS)
	$(QUIET)$(MAKE) -C src $(BIN)
.PHONY: src/$(BIN)

libtime/.git:
	git submodule update --init libtime

libtime/libtime.a: libtime/.git
	$(QUIET)$(MAKE) -C libtime libtime.a
.PHONY: libtime/libtime.a

ifndef NATIVE_C11
libc11/.git:
	git submodule update --init libc11

libc11/libc11.a: libc11/.git
	$(QUIET)$(MAKE) -C libc11 libc11.a
.PHONY: libc11/libc11.a
endif

