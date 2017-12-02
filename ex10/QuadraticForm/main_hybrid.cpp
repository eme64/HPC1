#include <iostream>
#include <mpi.h>

int main(int argc, char** argv)
{
    int n = 1024;
	
	int provided;
	MPI_Init(&argc, &argv);//_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
	
	int mpi_rank;
	int mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	//std::cout << mpi_rank <<", "<< mpi_size << " sign in" << std::endl;
	
	// we will subdivide along i, distribute the rows
	int n_local = n/mpi_size;
	int n_offset = mpi_rank * n_local;
	
	if(mpi_rank == mpi_size-1){
		n_local+= n % mpi_size; // rest of lines
	}
	
	
    double * const A = new double[n_local*n];
    double * const v = new double[n_local];
    double * const w = new double[n]; // need all !

	// -------------------- INIT DATA
	
	#pragma omp parallel
	{
	
		/// init A_ij = (i + 2.*j) / n^2
		const double fac0 = 1./(n*n);
		
		
		#pragma omp for nowait
		for (int i=0; i<n_local; ++i)
		    for (int j=0; j<n; ++j)
		        A[i*n + j] = ((i+n_offset) + 2.*j) * fac0;
		
		/// init v_i = 1. + 2. / (i+.5)
		#pragma omp for nowait
		for (int i=0; i<n_local; ++i)
		    v[i] = 1. + 2. / ((i+n_offset) + 0.5);
		
		/// init w_i = 1. - i / (3.*n)
		const double fac1 = 1./(3.*n);
		#pragma omp for nowait
		for (int j=0; j<n; ++j)
		    w[j] = 1. - j * fac1;
	
	}
	
	// -------------------- COMPUTE
	
    	double local_result = 0.;
    	
    	#pragma omp parallel for reduction(+: local_result)
    	for (int i=0; i<n_local; ++i)
        	for (int j=0; j<n; ++j)
            		local_result += v[i] * A[i*n + j] * w[j];
	
	// -------------------- REDUCE
	double result = 0.;
	
	MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
	if(mpi_rank == 0){
		std::cout << "Result = " << result << std::endl;
	}

    delete[] A;
    delete[] v;
    delete[] w;
	
	MPI_Finalize();
	
    return 0;
}
