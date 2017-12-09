# TODO 1: DEFINE TOOLS
CC = mpic++ -Wall -O3 -std=c++11

.PHONY: all
all: sheet

sheet: mainSheet.o VelocitySolverNSquared.o 
	# TODO 1: FILL IN COMPILE COMMAND HERE
	$(CC) $^ -o $@
%.o: %.cpp
	$(CC) -c $^ -o $@

.PHONY: clean
clean:
	rm -f *.o
	rm -f sheet
