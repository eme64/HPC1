#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <random>
// TODO 2: add header for OpenMP library
// ...
#include <omp.h>

// TODO 3: reimplement using corresponding OpenMP function
double get_wtime(void)		
{
	struct timeval t;

	gettimeofday(&t, NULL);
	return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}

inline double f(double x)	// function to integrate
{
	return 4.*sqrt(1.-x*x);
}

// WolframAlpha: integrate 4(1-x^2)^0.5 from 0 to 1 =	pi

int main(int argc, char *argv[])
{
	double a = 0.;            // integration range
	double b = 1.;
	unsigned long n = 1e8;    // number of samples

	double h = (b-a)/n;
	double ref = 3.14159265358979323846; 

	double t0 = get_wtime();

	// http://www.cplusplus.com/reference/random/uniform_real_distribution/
	//std::default_random_engine gen(314); // generator with given seed
	//std::uniform_real_distribution<double> uni(a,b);	

	// TODO 4: parallelize using OpenMP
	double res = 0.;
	int count_samples = 0;
	omp_set_num_threads(4);
	#pragma omp parallel reduction(+:res) reduction(+:count_samples)
	{
		printf("id: %d\n", omp_get_thread_num());
		double x;
		// init random variable:
		std::default_random_engine gen(314); // generator with given seed
		std::uniform_real_distribution<double> uni(a,b);
		
		for (unsigned long i = 0; i < n; i++) {
			x = uni(gen);
			res += f(x);
			count_samples += 1;
		}
		res *= h;
	}
	double t1 = get_wtime();

	printf("res:  %.16f\nref:  %.16f\nerror: %.5e\ntime: %lf s\nnum_samples: %d\nshould num samples: %lu\n", 
	res, ref, std::abs(res-ref), t1-t0, count_samples, n);

	return 0;
}
