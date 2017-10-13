CC = gcc # c
CXX = g++ # cpp
# INCLUDES ?= -I/cluster/apps/eigen/3.2.1/x86_64/gcc_4.8.2/serial/include/eigen3/

CFLAGS = -Wall -Wextra -std=c11 -lm -O3 -march=native
CPPFLAGS = -Wall -Wextra -std=c++11 -O3 -march=native
CPPFLAGS += -Wno-int-in-bool-context  # Eigen...

.DEFAULT_GOAL := diff
.PHONY: clean

OBJ1 = src/diff.o
OBJ2 = src/bench.o

diff: $(OBJ1)
	$(CC) $(CFLAGS) $(OBJ1) -o diff -lm
	
bench: $(OBJ2)
	$(CC) $(CFLAGS) $(OBJ2) -o bench -lm

# special target that generates an object file for a given source file (.cpp)
%.o: %.cpp
	$(CC) $(CXXFLAGS) -c $^ -o $@ -lm # have lm at end for linking !
	
# clean up target, called like this: make clean
clean:
	rm -f *~ diff bench $(OBJ1) $(OBJ2)

