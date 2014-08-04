include common.mk

BIN := nbody

all: $(BIN)

$(BIN): src/$(BIN)
	$(QUIET)$(CP) $< $@

distclean: clean

clean:
	$(MAKE) -C src clean
	$(MAKE) -C libtime clean
	$(MAKE) -C libc11 clean

src/$(BIN): libtime/libtime.a libc11/libc11.a
	$(QUIET)$(MAKE) -C src $(BIN)

libtime/libtime.a:
	$(QUIET)$(MAKE) -C libtime libtime.a

libc11/libc11.a:
	$(QUIET)$(MAKE) -C libc11 libc11.a

.PHONY: libc11/libc11.a libtime/libtime.a src/nbody
