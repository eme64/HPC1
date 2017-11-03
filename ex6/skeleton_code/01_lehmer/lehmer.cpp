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

    double ti2 = omp_get_wtime();
    std::cout << "Init time = " << ti2-ti1 << " seconds\n";

    // Study the function signature for the dsyev_() routine:
    // http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga442c43fca5493590f8f26cf42fed4044.html#ga442c43fca5493590f8f26cf42fed4044

    // required parameters for the dsyev_() routine
    char jobz = '?'/* TODO: configure dsyev */;
    char uplo = '?'/* TODO: configure dsyev */;

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

    lwork= (int) work[0];
    delete[] work;

    // prepape and issue the second (actual) call.
    work = new double[lwork];
    assert(work != NULL);

    double t1 = omp_get_wtime();

    // TODO: call dsyev_() here

    double t2 = omp_get_wtime();

    // Output results
    std::cout << "Elapsed time = " << t2-t1 << " seconds\n";
    std::cout << "Largest Eigenvalues:" << std::endl;

    // TODO: Report the 5 largest eigenvalues

    delete[] work;
    delete[] W;
    delete[] C;

    return 0;
}
