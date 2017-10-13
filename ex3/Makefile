all: mc_serial mc_omp

mc_serial: mc_serial.cpp
		g++ -O3 -Wall -std=c++11 mc_serial.cpp -o mc_serial

mc_omp: mc_omp.cpp
	g++ -O3 -Wall -std=c++11 -fopenmp mc_omp.cpp -o mc_omp

clean:
		rm -f mc_serial mc_omp

.PHONY: all clean
