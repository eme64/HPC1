#include <iostream>

int main(int argc, char** argv)
{
    int n = 1024;

    double * const A = new double[n*n];
    double * const v = new double[n];
    double * const w = new double[n];

#pragma omp parallel
    {
        /// init A_ij = (i + 2.*j) / n^2
        const double fac0 = 1./(n*n);
#pragma omp for nowait
        for (int i=0; i<n; ++i)
            for (int j=0; j<n; ++j)
                A[i*n + j] = (i + 2.*j) * fac0;

        /// init v_i = 1. + 2. / (i+.5)
#pragma omp for nowait
        for (int i=0; i<n; ++i)
            v[i] = 1. + 2. / (i + 0.5);

        /// init w_i = 1. - i / (3.*n)
        const double fac1 = 1./(3.*n);
#pragma omp for nowait
        for (int i=0; i<n; ++i)
            w[i] = 1. - i * fac1;
    }

    double result = 0.;
#pragma omp parallel for reduction(+: result)
    for (int i=0; i<n; ++i)
        for (int j=0; j<n; ++j)
            result += v[i] * A[i*n + j] * w[j];

    std::cout << "Result = " << result << std::endl;

    delete[] A;
    delete[] v;
    delete[] w;

    return 0;
}
