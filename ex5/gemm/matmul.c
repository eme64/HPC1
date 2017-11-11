#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <cblas.h>

#define N 512  // Number of rows and columns.


/*
 * Computes entries (i, j) of matrix C as dot products
 * "i-th row of A times j-th column of B"
 */
void __attribute__((optimize("O0")))  // Disable optimizations in GCC.
mm_mult_element_wise(double C[N*N], double A[N*N], double B[N*N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i*N+j] = 0.0;
            for (int k = 0; k < N; ++k) {
                C[i*N+j] += A[i*N+k] * B[k*N+j]; // "INSERT YOUR CODE HERE";
            }
        }
    }
}


/*
 * Calculate result matrix row by row.
 */
void mm_mult_row_wise(double C[N*N], double A[N*N], double B[N*N]) {
    for (int i = 0; i < N; ++i) {
        double Aik = A[i*N+0];
        for (int j = 0; j < N; ++j)
            C[i*N+j] = Aik * B[0*N+j];
        for (int k = 1; k < N; ++k) {
            Aik = A[i*N+k];
            for (int j = 0; j < N; ++j) {
                C[i*N+j] += Aik * B[k*N+j]; // "INSERT YOUR CODE HERE";
            }
        }
    }
}


/*
 * Calculate result matrix column by column.
 */
void mm_mult_column_wise(double C[N*N], double A[N*N], double B[N*N]) {
    for (int j = 0; j < N; ++j) {
        double Bkj = B[0*N+j];
        for (int i = 0; i < N; ++i)
            C[i*N+j] = A[i*N+0] * Bkj;
        for (int k = 1; k < N; ++k) {
            Bkj = B[k*N+j];
            for (int i = 0; i < N; ++i) {
                C[i*N+j] += A[i*N+k] * Bkj; // "INSERT YOUR CODE HERE";
            }
        }
    }
}

void mm_mult_parallel(double C[N*N], double A[N*N], double B[N*N]) {
    	
    	/*
    	#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		int id = omp_get_thread_num();
		
		
	}
	
	
	#pragma omp parallel reduction(+:sum)
	{
		
	}
	*/
	//omp_set_num_threads(8);
	
	/*
	#pragma omp parallel
	{
	#pragma omp single
	printf("num threads%d\n",omp_get_num_threads());
	}
	*/
	
	#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		double Aik = A[i*N+0];
		for (int j = 0; j < N; ++j)
			C[i*N+j] = Aik * B[0*N+j];
		for (int k = 1; k < N; ++k) {
			Aik = A[i*N+k];
			for (int j = 0; j < N; ++j) {
				C[i*N+j] += Aik * B[k*N+j];
			}
		}
        }
    
}

void mm_mult_blas(double C[N*N], double A[N*N], double B[N*N]) {
    // openblas_set_num_threads(1);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    			N,N,N,
    			1.0, A, N,
    			B, N, 0.0,
    			C, N);
    // https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm
    // A m*k
    // B k*n
    // C m*n
    // C=alpha*A*B+beta*C
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
}

/*
 * Return wall time in seconds.
 */
double get_wtime(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}


/*
 * Benchmark execution time for the given matrix multiplication function.
 */
double benchmark(void (*func)(double C[N*N], double A[N*N], double B[N*N]),
                 double C[N*N],
                 double A[N*N],
                 double B[N*N]) {
    /*
     * Run the given function `func` 10 times with arguments C, A and B (target
     * matrix and two input matrices), measure execution time using `get_wtime`
     * and return the fastest time.
     */

    /* INSERT YOUR CODE HERE */
    double time_min = 1e10;  // Best execution time.
    for (int trial = 0; trial < 10; ++trial) {
        double time = get_wtime();

        func(C, A, B);

        time = get_wtime() - time;
        if (time < time_min)
            time_min = time;
    }
    return time_min;
}


double A[N*N], B[N*N];
double C1[N*N], C2[N*N], C3[N*N], C4[N*N], C5[N*N];

int main(void) {
    /* Fill matrices a and b with random values. */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i*N+j] = (double)rand() / (double)RAND_MAX;
            B[i*N+j] = (double)rand() / (double)RAND_MAX;
        }
    }

    /* Element-wise computation. */
    const double t1 = benchmark(mm_mult_element_wise, C1, A, B);
    printf("element-wise computation took t1 = %6.1lfms\n", 1000 * t1);

    /* Row-wise computation. */
    const double t2 = benchmark(mm_mult_row_wise, C2, A, B);
    printf("row-wise computation took     t2 = %6.1lfms\n", 1000 * t2);

    /* Column-wise computation. */
    const double t3 = benchmark(mm_mult_column_wise, C3, A, B);
    printf("column-wise computation time  t3 = %6.1lfms\n", 1000 * t3);
    
    /* parallel computation. */
    const double t4 = benchmark(mm_mult_parallel, C4, A, B);
    printf("parallel computation time  t4 = %6.1lfms\n", 1000 * t4);
    
    /* blas computation. */
    const double t5 = benchmark(mm_mult_blas, C5, A, B);
    printf("blas computation time  t5 = %6.1lfms\n", 1000 * t5);
    

    /* Error calculation. */
    double diff;
    double error2 = 0.0, mxerr2 = 0.0;
    double error3 = 0.0, mxerr3 = 0.0;
    double error4 = 0.0, mxerr4 = 0.0;
    double error5 = 0.0, mxerr5 = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            error2 += (C2[i*N+j] - C1[i*N+j]) * (C2[i*N+j] - C1[i*N+j]);
            error3 += (C3[i*N+j] - C1[i*N+j]) * (C3[i*N+j] - C1[i*N+j]);
            error4 += (C4[i*N+j] - C1[i*N+j]) * (C4[i*N+j] - C1[i*N+j]);
            error5 += (C5[i*N+j] - C1[i*N+j]) * (C5[i*N+j] - C1[i*N+j]);
            diff   = fabs(C2[i*N+j] - C1[i*N+j]);
            mxerr2 = diff > mxerr2 ? diff : mxerr2;
            diff   = fabs(C3[i*N+j] - C1[i*N+j]);
            mxerr3 = diff > mxerr3 ? diff : mxerr3;
            
            diff   = fabs(C4[i*N+j] - C1[i*N+j]);
            mxerr4 = diff > mxerr4 ? diff : mxerr4;
            
            diff   = fabs(C5[i*N+j] - C1[i*N+j]);
            mxerr5 = diff > mxerr5 ? diff : mxerr5;
        }
    }

    error2 = sqrt(error2) / (double)N;
    error3 = sqrt(error3) / (double)N;
    printf("RMS error for %d x %d row-wise matrix mult. = %15.8e\n", N, N, error2);
    printf("max error for %d x %d row-wise matrix mult. = %15.8e\n", N, N, mxerr2);
    printf("RMS error for %d x %d column-wise matrix mult. = %15.8e\n", N, N, error3);
    printf("max error for %d x %d column-wise matrix mult. = %15.8e\n", N, N, mxerr3);
    
    printf("RMS error for %d x %d parallel matrix mult. = %15.8e\n", N, N, error4);
    printf("max error for %d x %d parallel matrix mult. = %15.8e\n", N, N, mxerr4);
    
    printf("RMS error for %d x %d blas matrix mult. = %15.8e\n", N, N, error5);
    printf("max error for %d x %d blas matrix mult. = %15.8e\n", N, N, mxerr5);
    

    return 0;
}
