config ?= debug

ifeq ($(shell uname -s), Darwin)
CC=gcc-7
else
CC=gcc
endif


CPPFLAGS= -std=c++11 -fopenmp
LIBS= -lstdc++ -lm
CPPFLAGS+= -fstrict-aliasing -mtune=native  -march=native
CPPFLAGS+= -fmerge-all-constants -fassociative-math

ifeq "$(config)" "debug"
CPPFLAGS+= -g -O0
#CPPFLAGS+= -fsanitize=address
#LIBS += -fsanitize=address
endif

ifeq "$(config)" "prod"
CPPFLAGS += -DNDEBUG -O3 -ffast-math
endif


CPPFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wpedantic
CPPFLAGS+= -Wmissing-declarations -Wredundant-decls -Wshadow -Wwrite-strings
CPPFLAGS+= -Woverloaded-virtual -Wno-unused-parameter


exec_oja: main_oja.o
	$(CC) $(CPPFLAGS) $(LIBS) main_oja.o -o $@

exec_backprop: main_backprop.o
	$(CC) $(CPPFLAGS) $(LIBS) main_backprop.o -o $@

all: exec_oja exec_backprop
.DEFAULT_GOAL := all

%.o: %.cpp
	$(CC) $(CPPFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.dSYM *.s *.d exec_*
