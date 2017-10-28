#include <stdio.h>
#include <stdlib.h>
#include <iostream>

int main(int argc, char** argv)
{
	int n = 1024;

	// allocate memory
	double *A = (double *)malloc(n*n*sizeof(double));
	double *v = (double *)malloc(n*sizeof(double));
	double *w = (double *)malloc(n*sizeof(double));

	/// init A_ij = (i + 2.*j) / n^2
	for (int i=0; i<n; ++i)
		for (int j=0; j<n; ++j)
			A[i*n+j] = (i + 2.*j) / (n*n);
    
	/// init v_i = 1. + 2. / (i+.5)
	for (int i=0; i<n; ++i)
		v[i] = 1. + 2. / (i + 0.5);
    
	/// init w_i = 1. - i / (3.*n)
	for (int i=0; i<n; ++i)
		w[i] = 1. - i / 3. / n;


	/// compute
	double sum = 0;
	for(int i=0; i<n; i++)
		for(int j=0; j<n; j++)
			sum=sum+ v[i]*A[i*n+j]*w[j];
	
	std::cout << "result: " << sum << std::endl;	
		
	/// free memory
	free(A);
	free(v);
	free(w);

	return 0;
}
