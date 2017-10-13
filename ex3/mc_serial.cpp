#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <random>


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
	std::default_random_engine gen(314); // generator with given seed
	std::uniform_real_distribution<double> uni(a,b);	

	double res = 0.;
  double x;
	for (unsigned long i = 0; i < n; i++) {
		x = uni(gen);
		res += f(x);
	}
	res *= h;

	double t1 = get_wtime();

	printf("res:  %.16f\nref:  %.16f\nerror: %.5e\ntime: %lf s\n", 
      res, ref, std::abs(res-ref), t1-t0);

	return 0;
}
