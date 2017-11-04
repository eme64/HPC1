// File       : pca.cpp
// Created    : Thu Nov 02 2017 03:01:30 PM (+0100)
// Description: Principal component analysis applied to image compression
// Copyright 2017 ETH Zurich. All Rights Reserved.
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include <omp.h>
#include <zlib.h>

// interface for LAPACK routines.  On Euler, you must load the MKL library:
// $ module load mkl
#include <mkl_lapack.h>


///////////////////////////////////////////////////////////////////////////////
// helpers
double *read_gzfile(char *filename, int frows, int fcols, int rows, int cols)
{
    double *A, *buf;
    gzFile fp;
    int i;

    A = new (std::nothrow) double[rows*cols];
    assert(A != NULL);

    buf = new (std::nothrow) double[fcols];
    assert(buf != NULL);

    fp = gzopen(filename, "rb");
    if (fp == NULL) {
        std::cout << "Input file not available!\n";
        exit(1);
    }

    for (i = 0; i < rows; i++) {
        gzread(fp, buf, fcols*sizeof(double));
        memcpy(&A[i*cols], buf, cols*sizeof(double));
    }
    gzclose(fp);

    delete[] buf;

    return A;
}

void write_ascii(const char* const filename, const double* const data, const int rows, const int cols)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        std::cout << "Failed to create output file\n";
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fp, "%.4lf ", data[i*cols+j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//
// elvis.bin.gz:   469x700
// cyclone.bin.gz: 4096x4096
// earth.bin.gz:   9500x9500
//
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // input parameters (default)
    int m = 469, n = 700;                        // image size (rows, columns)
    int npc = 50;                                // number of principal components
    char *inp_filename = (char *)"../../data/elvis.bin.gz"; // input filename (compressed binary)
    char *out_filename = (char *)"../../data/elvis.50.bin"; // output filename (text)

    // parse input parameters
    if ((argc != 1) && (argc != 11)) {
        std::cout << "Usage: " << argv[0] << " -m <rows> -n <cols> -npc <number of principal components> -if <input filename> -of <output filename>\n";
        exit(1);
    }

    for(int i = 1; i < argc; i++ ) {
        if( strcmp( argv[i], "-m" ) == 0 ) {
            m = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-n" ) == 0 ) {
            n = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-npc" ) == 0 ) {
            npc = atoi(argv[i+1]);
            i++;
        }
        if( strcmp( argv[i], "-if" ) == 0 ) {
            inp_filename = argv[i+1];
            i++;
        }
        if( strcmp( argv[i], "-of" ) == 0 ) {
            out_filename = argv[i+1];
            i++;
        }
    }

    if (npc > n) npc = n;
    double t_elapsed;

    ///////////////////////////////////////////////////////////////////////////
    // Read image data.  The image dimension is m x n.  The returned pointer
    // points to the data in row-major order.  That is, if (i,j) corresponds to
    // to the row and column index, respectively, you access the data with
    // pixel_{i,j} = I[i*n + j], where 0 <= i < m and 0 <= j < n.
    ///////////////////////////////////////////////////////////////////////////
    double *I = read_gzfile(inp_filename, m, n, m, n);

    // A = transpose(I), so image features (columns) are stored in rows.  More
    // efficient to work with the data in this layout.
    double *A = new (std::nothrow) double[n*m];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i*m + j] = I[j*n+i];
        }
    }
   	for(int i=0;i<10; i++){
                std::cout << "A[" << i << "] = " << A[i] << std::endl;
        } 

    ///////////////////////////////////////////////////////////////////////////
    // TODO: Implement your PCA algorithm here
    // 1. Compute mean and standard deviation of your image features
    // 2. Normalize the data
    // 3. Build the covariance matrix
    // 4. Compute the eigenvalues and eigenvectors of the covariance matrix.
    //    Use LAPACK here.
    // 5. Compute the principal components and report the compression ratio
    // 6. Reconstruct the image from the compressed data and dump the image in
    //    ascii.
    ///////////////////////////////////////////////////////////////////////////
    double start_t = omp_get_wtime();

    ///////////////////////////////////////////////////////////////////////////
    // TODO: 1.
    t_elapsed = -omp_get_wtime();

    double *AMean = new (std::nothrow) double[n];
    double *AStd  = new (std::nothrow) double[n];
    assert(AMean != NULL);
    assert(AStd  != NULL);
	
	
    for (int i = 0; i < n; i++)
    {
        // TODO: compute mean and standard deviation of features of A
        AMean[i]=0;
	for(int j=0; j<m; j++){
		AMean[i]+=A[i*m+j];
	}
	AMean[i] /= m; // mean
	
		
	AStd[i]=0;
	for(int j=0;j<m;j++){// sum of squares
		AStd[i]+= (A[i*m+j]-AMean[i])*(A[i*m+j]-AMean[i]);
	}
	AStd[i] = std::sqrt(AStd[i]/(m-1)); // estimated std deviation
    }
    t_elapsed += omp_get_wtime();
    std::cout << "MEAN/STD TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // TODO: 2.
    t_elapsed = -omp_get_wtime();

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            // TODO: normalize data here
            	A[i*m+j]-= AMean[i]; // centralize
		A[i*m+j]/= AStd[i]; // normalize
        }
    }
	
	for(int i=0;i<10; i++){
                std::cout << "AMean[" << i << "] = " << AMean[i] << std::endl;
		std::cout << "AStd[" << i << "] = " << AStd[i] << std::endl;
		std::cout << "B[" << i << "] = " << A[i] << std::endl;
        }
	
    t_elapsed += omp_get_wtime();
    std::cout << "NORMAL. TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // TODO: 3.
    t_elapsed = -omp_get_wtime();
    double *C = new (std::nothrow) double[n*n];
    assert(C!=NULL);

    // TODO: Compute covariance matrix here
	
	for(int i=0;i<n;i++){// doing full matrix, could only do upper half
		for(int j=0; j<n;j++){
			C[i*n + j] = 0; // dot of i'th and j'th row of C
			for(int k=0; k<m; k++){
				C[i*n+j]+= A[i*m+k] * A[j*m+k];
			}
			C[i*n+j]/=(double)(m-1);// divide matrix
		}
	}
	for(int i=0;i<10; i++){
		std::cout << "C[" << i << "] = " << C[i] << std::endl;
	}
    t_elapsed += omp_get_wtime();
    std::cout << "C-MATRIX TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////
    // TODO 4. LAPACK
    t_elapsed = -omp_get_wtime();

    // see also for the interface to dsyev_():
    // http://www.netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga442c43fca5493590f8f26cf42fed4044.html#ga442c43fca5493590f8f26cf42fed4044
    char jobz = 'V'; // TODO: compute both, eigenvalues and orthonormal eigenvectors
    char uplo = 'U'; // TODO: how did you compute the (symmetric) covariance matrix?
    int info, lwork;

    double *W = new (std::nothrow) double[n]; // eigenvalues
    assert(W != NULL);

    double *work = new (std::nothrow) double[2];
    assert(work != NULL);

    // first call to dsyev_() with lwork = -1 to determine the optimal
    // workspace (cheap call)
    lwork = -1;

    // TODO: call dsyev here
	dsyev_(&jobz, &uplo, &n, C, &n, W, work, &lwork, &info);
    lwork = (int)work[0];
    delete[] work;

    // allocate optimal workspace
    work = new (std::nothrow) double[lwork];
    assert(work != NULL);

    // second call to dsyev_(), eigenvalues and eigenvectors are computed here
    // TODO: call dsyev here
	dsyev_(&jobz, &uplo, &n, C, &n, W, work, &lwork, &info);
    t_elapsed += omp_get_wtime();
    std::cout << "DSYEV TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////
	std::cout << "exit status: info = " << info << std::endl;
	std::cout << "EVec of biggest EVal:"<< std::endl;
	for(int i=0;i<10;i++){
		std::cout << "V[" << i << "]= " << C[i+n*(n-1)]<< std::endl;
	}
	for(int i=0;i<10;i++){
                std::cout << "D[" << n-i-1 << "]= " << W[n-i-1]<< std::endl;
        }
    ///////////////////////////////////////////////////////////////////////////
    // TODO: 5.
    t_elapsed = -omp_get_wtime();
    double *PCReduced = new (std::nothrow) double[m*npc];
    assert(PCReduced != NULL);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < npc; j++)
        {
            // TODO: compute the principal components
            // C now holds the eigenvectors, stored row-wise (wrong ?)
            	PCReduced[i*npc + j]=0;
		for(int k=0;k<n;k++){
			PCReduced[i*npc + j] += A[k*m + i]*C[(j+n-npc)*n+k];
		}
        }
    }
	std::cout << "last line of PCReduced:" << std::endl;
	for(int i=0;i<10;i++){
                std::cout << "PCReduced["<<i<<"]= " << PCReduced[i+(m-1)*npc] << std::endl;
        }

    // TODO: Report the compression ratio
	std::cout << std::endl << "------- compression ratio:" << std::endl;
	std::cout << "data in: A = m*n = " << m << "*" << n << " = " << m*n << std::endl;
	std::cout << "data out: PCReduced + VReduced + AStd + AMean" << std::endl;
	std::cout << "= m*npc + n*npc + n + n" << std::endl;
	std::cout << "= " << m << "*" << npc << " + " << n << "*" << npc << " + 2*" << n << std::endl;
	std::cout << "ratio: out/in = " << (double)(m*npc + n*npc + 2*n)/(double)(m*n) << std::endl << std::endl;

    t_elapsed += omp_get_wtime();
    std::cout << "PCREDUCED TIME=" << t_elapsed << " seconds\n";
    ///////////////////////////////////////////////////////////////////////////

    double end_t = omp_get_wtime();
    std::cout << "OVERALL TIME=" << end_t - start_t << " seconds\n";

    ///////////////////////////////////////////////////////////////////////////
    // TODO: 6
    double *Z = new (std::nothrow) double[m*n]; // memory for reconstructed image
    assert(Z != NULL);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // TODO: Reconstruct image here.  Don't forget to denormalize.  The
            // dimension of the reconstructed image is m x n (rows x columns).
            // Z[i*n + j] = ...
            Z[i*n + j] = 0;
		for(int k=0; k<npc;k++){
			//PCReduced[i*npc + j] += A[k*m + i]*C[(j+n-npc)*n+k];
			Z[i*n+j] += PCReduced[i*npc+k]*C[j+n*(k+n-npc)];
		}
		Z[i*n+j] *= AStd[j];
		Z[i*n+j] += AMean[j];
        }
    }
	for(int i=0;i<10;i++){
		std::cout << "Z["<<i<<"]= " << Z[i] << std::endl;
	}


    // Write the reconstructed image in ascii format.  You can view the image
    // in Matlab with the show_image.m script.
    write_ascii(out_filename, Z, m, n);
    
	// doing some error calculations:
	double error = 0;
	
	for (int i = 0; i < n; i++) {
        	for (int j = 0; j < m; j++) {
            		error += (I[j*n+i]-Z[j*n+i])*(I[j*n+i]-Z[j*n+i]);
        	}
    	}
	
	std::cout << " - - ERROR: " << error/(double)(m*n) << std::endl;
	
	delete[] I; // finally releasing input file

	///////////////////////////////////////////////////////////////////////////

    // cleanup
    delete[] work;
    delete[] W;
    delete[] C;
    delete[] Z;
    delete[] PCReduced;
    delete[] A;
    delete[] AMean;
    delete[] AStd;

    return 0;
}
