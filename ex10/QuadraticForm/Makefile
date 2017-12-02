CXX = g++
MPICXX = mpic++

CPPFLAGS = -Wall -Wextra -O3 -fopenmp

.PHONY: clean

all: main_shared main_distributed main_hybrid

clean:
	rm -f *~ main_shared main_distributed main_hybrid

main_shared: main_shared.cpp
	$(CXX) $(CPPFLAGS) main_shared.cpp -o main_shared
	
main_distributed: main_distributed.cpp
	$(MPICXX) $(CPPFLAGS) main_distributed.cpp -o main_distributed
	
main_hybrid: main_hybrid.cpp
	$(MPICXX) $(CPPFLAGS) main_hybrid.cpp -o main_hybrid
