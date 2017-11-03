// File       : lehmer.cpp
// Created    : Thu Nov 02 2017 05:40:48 PM (+0100)
// Description: Compute eigenvalues of Lehmer matrix
// Copyright 2017 ETH Zurich. All Rights Reserved.
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <omp.h>
#include <algorithm>
// interface for LAPACK routines.  On Euler, you must load the MKL library:
// $ module load mkl
#include <mkl_lapack.h>


///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int N = 8;  // default matrix size (NxN)

    for(int i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-N" ) == 0 ) {
            N = atoi(argv[i+1]);
            i++;
        }
    }

    // Lehmer matrix
    int rows = N, cols = N;
    double *C = new (std::nothrow) double[rows*cols];

    // matrix initialization
    double ti1 = omp_get_wtime();

    // TODO: Initialize Lehmer matrix
	
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			C[i*N+j] = (double)std::min(i+1,j+1)/(double)std::max(i+1,j+1);
			//std::cout << C[i*N+j] << ", ";
		}
		//std::cout << ":" << std::endl;
	}
	
	//std::cout << C[2*N+3] << std::endl;

    double ti2 = omp_get_wtime();
    std::cout << "Init time = " << ti2-ti1 << " seconds\n";

    // Study the function signature for the dsyev_() routine:
    // http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga442c43fca5493590f8f26cf42fed4044.html#ga442c43fca5493590f8f26cf42fed4044
	
	/*
 	I am working from this example code:
	https://userinfo.surfsara.nl/systems/lisa/software/lapacke
	*/

    // required parameters for the dsyev_() routine
    char jobz = 'V'; // eventually we want also the vectors
    char uplo = 'U'; // I guess 'L' works just fine too given the data initialization

    // first call to dsyev_() with lwork = -1 to determine the optimal
    // workspace.  This does not yet compute the eigenvalues but determines the
    // optimal workset.
    double *work, *W;
    W = new double[N];
    assert(W != NULL);
    work = new double[1];
    assert( work != NULL);

    int info, lwork;
    lwork=-1;

    // TODO: call dsyev_() here
	dsyev_(&jobz, &uplo, &N, C, &N, W, work, &lwork, &info);
	// not quite sure with the lda input...
	//std::cout << "info: " << info << std::endl;
    lwork= (int) work[0]+1;//+1 because fortran???
	//std::cout << "optimal lwork: "<< lwork << std::endl;
    delete[] work;

    // prepape and issue the second (actual) call.
    work = new double[lwork];
    assert(work != NULL);

    double t1 = omp_get_wtime();

    // TODO: call dsyev_() here
	dsyev_(&jobz, &uplo, &N, C, &N, W, work, &lwork, &info);
	//std::cout << "info: " << info << std::endl;
    double t2 = omp_get_wtime();

    // Output results
    std::cout << "Elapsed time = " << t2-t1 << " seconds\n";
    std::cout << "Largest Eigenvalues:" << std::endl;

    // TODO: Report the 5 largest eigenvalues
	
	for(int i=1; i<=5; i++){
		std::cout << i << ": " << W[N-i] << std::endl;
	}

    delete[] work;
    delete[] W;
    delete[] C;

    return 0;
}
