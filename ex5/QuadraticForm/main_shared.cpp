#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>

int main(int argc, char** argv)
{
	int n = 1024;
	
	
	// allocate memory
	double *A = (double *)malloc(n*n*sizeof(double));
	double *v = (double *)malloc(n*sizeof(double));
	double *w = (double *)malloc(n*sizeof(double));

	/// compute
	/*    -------------------------------- sequential
	double sum = 0;
	for(int i=0; i<n; i++)
		for(int j=0; j<n; j++)
			sum=sum+ v[i]*A[i*n+j]*w[j];
	
	std::cout << "result: " << sum << std::endl;	
	*/
	
	/*
	 -------------- some assumptions and consequences:
	 
	 A will not fit in cache, but v or w can. (here cache of numa-node, not per processor)
	 
	 This means we want to only load parts of A per node,
	 v and w have to be loaded from one node to the others once each.
	 
	 Let's assume we partition along i (j makes not much sense):
	 Each node still needs all of w.
	 But we can load only parts of v and parts of A per node.
	 We could make copies of w for each node, but that does not help if it fits in the cache anyway and is never modified.
	 
	 With the first touch policy we can allocate the memory anywhere,
	 but we have to pay attention what we initiallize where.
	 These blocks should be alligned with page-size, otherwise we do not gain as much.
	 If n is big and base-2 then A will not be a problem. Pages are 4K usually. One line/row of A is 8*1K.
	 If n is fairly small and #nodes big it might happen that some pages will be shared.
	 However on Euler using 2 nodes and having v be 8K it will fall into exactly 2 pages of 4K.
	 
	 Ok, let's get to work.	
	*/
	
	// -------------------------------- w needed in full by all -> init anywhere
	/// init w_i = 1. - i / (3.*n)
	for (int i=0; i<n; ++i)
		w[i] = 1. - i / 3. / n;
	
	// -------------------------------- init A and v split into regions
	#pragma omp parallel
	{
		int num_threads = omp_get_num_threads();
		
		int id = omp_get_thread_num();
		
		int i_from = n*id/num_threads;
		int i_to = n*(id+1)/num_threads;
		/*
		#pragma omp critical
                {
                        std::cout << "id: " << id << std::endl;
                	std::cout << "from " << i_from << " to " << i_to << std::endl;
		}
		*/
		/// init A_ij = (i + 2.*j) / n^2
		for (int i=i_from; i<i_to; ++i)
			for (int j=0; j<n; ++j)
				A[i*n+j] = (i + 2.*j) / (n*n);
	    	
		/// init v_i = 1. + 2. / (i+.5)
		for (int i=i_from; i<i_to; ++i)
			v[i] = 1. + 2. / (i + 0.5);
	}
	// as far as I know this implicit barrier is not needed, but I'll leave it anyway
	// this way we certainly have all data initiallized before any calculation.
	
	// -------------------------------- go calculate
	double sum = 0;
	
	#pragma omp parallel reduction(+:sum)
	{
		int num_threads = omp_get_num_threads();
		int id = omp_get_thread_num();
		//std::cout << "id: " << id << std::endl;
		
		int i_from = n*id/num_threads;
                int i_to = n*(id+1)/num_threads;
		/*
		#pragma omp critical
                {
                        std::cout << "id: " << id << std::endl;
                        std::cout << "from " << i_from << " to " << i_to << std::endl;
                }
		*/
		for(int i=i_from; i<i_to; i++)
			for(int j=0; j<n; j++)
				sum=sum+ v[i]*A[i*n+j]*w[j];
	}
	
	std::cout << "sum: " << sum << std::endl;
		
	/// free memory
	free(A);
	free(v);
	free(w);

	return 0;
}
