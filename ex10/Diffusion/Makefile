# Makefile for HPCSE I (2016HS) Exam, 23.12.2015
# Prof. M. Troyer, Dr. P. Hadjidoukas
# Coding 2 : Diffusion Statistics
#
# Task: complete the compile commands

MPICXX=mpic++

CXXFLAGS=-Wall -O3 -std=c++11

all: diffusion2d_mpi

diffusion2d_mpi: diffusion2d_mpi.cpp timer.hpp
	$(MPICXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	-rm -f diffusion2d_mpi density_mpi.dat
