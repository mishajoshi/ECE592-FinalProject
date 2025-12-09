CC = gcc
CFLAGS = -O3 -march=native -Wall

all: gemm_victim

gemm_victim: gemm_victim.c
	$(CC) $(CFLAGS) -o gemm_victim gemm_victim.c

clean:
	rm -f gemm_victim
