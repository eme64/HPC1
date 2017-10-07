#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define N 100000  // Number of rows and columns.
#define M_PI 3.14159265358979323846
#define MIN(a,b) (((a)<(b))?(a):(b))

/* 
 * Computes entries (i, j) of matrix C as dot products
 * "i-th row of A times j-th column of B"
 */
void
//__attribute__((optimize("O0")))  // Disable optimizations in GCC.
time_step_euler(double u_n[N], double u_np1[N], double alpha, double dt, double dx) {
    double fac = (dt*alpha)/(dx*dx);
    
    for (int i = 1; i < N-1; ++i) {
        u_np1[i] = u_n[i] + fac*(u_n[i-1]-2.0*u_n[i]+u_n[i+1]);
    }
    u_np1[0] = u_np1[1]; // nonreflecting BC
    u_np1[N-1] = u_np1[N-2];
}


double get_wtime(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}


double benchmark() {
     
     // initialize u0
     double L = 1000.0;
     double dx = L/(N-1);
     double alpha = 1.0e-4;
     double dt = dx*dx/(2.0*alpha);
     
     double u0[N];
     double u1[N];
     
     for(int i=0; i<N; i++){
     	u0[i] = sin(2.0*M_PI*dx*i/L);
     	u1[i] = 0.0;
     }
     
     double mintime = 10000000.0; // choose some ridiculously high value
     
     for(int i=0; i<1000; i++){ // get better minimum
     	double start = get_wtime();
     	
     	time_step_euler(u0, u1, alpha, dt, dx); // call
     	
     	double diff = get_wtime()- start;
     	
     	if (i==0) {mintime = diff;} else {mintime = MIN(diff, mintime);}
     }
     
     return mintime;
}


int main(void) {
    
    const double t1 = benchmark();
    printf("computation took t1 = %6.3lfms\n", 1000 * t1);
    
    return 0;
}

