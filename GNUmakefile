include common.mk

BIN := nbody

all: $(BIN)

$(BIN): src/$(BIN)
	$(QUIET)$(CP) $< $@

distclean: clean

clean:
	$(RM) $(BIN)
	$(MAKE) -C src clean
	$(MAKE) -C libtime clean
	$(MAKE) -C libc11 clean

src/$(BIN): libtime/libtime.a libc11/libc11.a
	$(QUIET)$(MAKE) -C src $(BIN)

libtime/.git:
	git submodule update --init libtime

libc11/.git:
	git submodule update --init libc11

libtime/libtime.a: libtime/.git
	$(QUIET)$(MAKE) -C libtime libtime.a

libc11/libc11.a: libc11/.git
	$(QUIET)$(MAKE) -C libc11 libc11.a

.PHONY: libc11/libc11.a libtime/libtime.a src/nbody
