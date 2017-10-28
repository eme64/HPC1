ALL= matmul

all: $(ALL)

.PHONY: clean
clean:
	-rm -f $(ALL)



matmul: matmul
	gcc -O3 -o matmul matmul.c -I ~/OPENBLAS/include/ -L~/OPENBLAS/lib -lopenblas -fopenmp -lm
