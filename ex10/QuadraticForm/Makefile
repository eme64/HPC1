CXX = g++-7

CPPFLAGS = -Wall -Wextra -O3 -fopenmp

.PHONY: clean

all: main_shared

clean:
	rm -f *~ main_shared

main_shared: main_shared.cpp
	$(CXX) $(CPPFLAGS) main_shared.cpp -o main_shared
