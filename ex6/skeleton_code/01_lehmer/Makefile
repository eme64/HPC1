# Make sure to load: gcc, mkl

CC = g++
CFLAGS = -O3 -Wall

LIBPATH=$(MKLROOT)/lib/intel64
MKL_SERIAL_LINK=-Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
MKL_PARA_LINK=-Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
LDFLAGS = -lm -lpthread -lz -ldl -fopenmp

all: lehmer_seq lehmer_omp

lehmer_seq: lehmer_seq.o
	$(CC) -o $@ $@.o $(OBJECTS) $(LDFLAGS) $(MKL_SERIAL_LINK) $(LDFLAGS)

lehmer_seq.o: lehmer.cpp
	$(CC) $(CFLAGS) -o lehmer_seq.o -c lehmer.cpp

lehmer_omp: lehmer_omp.o
	$(CC) -o $@ $@.o $(OBJECTS) $(LDFLAGS) $(MKL_PARA_LINK) $(LDLFLAGS)

lehmer_omp.o: lehmer.cpp
	$(CC) $(CFLAGS) -fopenmp -o lehmer_omp.o -c lehmer.cpp

clean:
	rm -f *.o *~ lehmer_seq lehmer_omp

