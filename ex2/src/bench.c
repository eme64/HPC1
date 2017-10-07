#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define N 100000  // Number of steps
#define M_PI 3.14159265358979323846
#define MIN(a,b) (((a)<(b))?(a):(b))



double get_wtime(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}


double benchmark(int n) {
     
     double mintime = 10000000.0; // choose some ridiculously high value
     
     for(int i=0; i<100; i++){ // get better minimum
     	
     	
     	float A[n];
     	float B[n];
     	
     	for (int j=0; j<n; j++){
     		A[j] = 0;
     		B[j] = 1.0e-10*(j+1.0);
     	}
     	
     	double start = get_wtime();
     	
     	for(int k=0; k<N; k++){
	     	for (int j=0; j<n; j++){
	     		B[j] = B[j]*1.1;
	     		A[j] = A[j] + B[j];
	     	}
	}
	
	
	/*
	float A[n];
	
	for (int j=0; j<n; j++){
     		A[j] = 1.0e-10*(j+1.0);
     	}
	
	double start = get_wtime();
	
	for(int k=0; k<N; k++){
	     	for (int j=0; j<n; j++){
	     		A[j] = A[j] + 1.0e-12*j;
	     	}
	}
	*/
     	
     	double diff = get_wtime()- start;
     	
     	if (i==0) {mintime = diff;} else {mintime = MIN(diff, mintime);}
     }
     
     return mintime;
}


int main(void) {
    
    for(int n=2; n<3000; n=n*2){
    	const double t1 = benchmark(n);
    	double performance = 2*N*n/(t1*1000000000.0);
    	printf("[n] = %d, performance: %6.3lf GF/s ,computation took t1 = %6.3lfms\n", n, performance, 1000 * t1);
    }
    
    return 0;
}

