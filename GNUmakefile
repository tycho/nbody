include common.mk

all: nbody

nbody: src/nbody
	$(QUIET)$(CP) $< $@

clean:
	$(MAKE) -C src clean
	$(MAKE) -C libtime clean
	$(MAKE) -C libc11 clean

src/nbody: libtime/libtime.a libc11/libc11.a
	$(QUIET)$(MAKE) -C src nbody

libtime/libtime.a:
	$(QUIET)$(MAKE) -C libtime libtime.a

libc11/libc11.a:
	$(QUIET)$(MAKE) -C libc11 libc11.a

.PHONY: libc11/libc11.a libtime/libtime.a src/nbody
