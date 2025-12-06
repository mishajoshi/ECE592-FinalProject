CC=gcc
CFLAGS=-O2 -Wall -Wextra -std=c11
LDFLAGS=-pthread

TARGETS=cache_probe tlb_probe btb_probe pht_probe

all: $(TARGETS)

cache_probe: cache_probe.c common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

tlb_probe: tlb_probe.c common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

btb_probe: btb_probe.c common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

pht_probe: pht_probe.c common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean all
clean:
	rm -f $(TARGETS)
