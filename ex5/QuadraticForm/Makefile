ALL=skeleton shared

all: $(ALL)

.PHONY: clean
clean:
	-rm -f $(ALL)

skeleton: skeleton.cpp
	g++ skeleton.cpp -o $@

shared: main_shared.cpp
	g++ -O3 -Wall -std=c++11 -fopenmp main_shared.cpp -o main_shared
