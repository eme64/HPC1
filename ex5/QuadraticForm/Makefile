ALL=skeleton shared

all: $(ALL)

.PHONY: clean
clean:
	-rm -f $(ALL)

skeleton: skeleton.cpp
	g++ skeleton.cpp -o $@

shared: main_shared.cpp
	g++ -fopenmp main_shared.cpp -o shared
