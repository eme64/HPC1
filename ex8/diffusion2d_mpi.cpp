#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include "timer.hpp"

#define DIAGNOSTICS_TAG 111
#define DATALINE_TAG 112

struct Diagnostics
{
    double time;
    double heat;

    Diagnostics(double time, double heat) : time(time), heat(heat) {}
};

class Diffusion2D_MPI {
public:
    Diffusion2D_MPI(const double D,
                    const double L,
                    const int N,
                    const double dt,
                    const int rank,
                    const int procs)
    : D_(D), L_(L), N_(N), dt_(dt), rank_(rank), procs_(procs)
    {
        // Real space grid spacing.
        dr_ = L_ / (N_ - 1);

        // Stencil factor.
        fac_ = dt_ * D_ / (dr_ * dr_);

        // Number of rows per process.
        local_N_ = N_ / procs_;

        // Small correction for the last process.
        if (rank_ == procs - 1)
            local_N_ += N_ % procs_;

        // Actual dimension of a row (+2 for the ghosts).
        real_N_ = N_ + 2;
        Ntot_ = (local_N_ + 2) * (N_ + 2);

        rho_.resize(Ntot_, 0.0);
        rho_tmp_.resize(Ntot_, 0.0);

        // Check that the timestep satisfies the restriction for stability.
        if (rank_ == 0) {
            std::cout << "timestep from stability condition is "
                      << dr_ * dr_ / (4. * D_) << '\n';
        }

        initialize_density();
    }
	
    void advance()
    {
        MPI_Request req[4];
        MPI_Status status[4];

        int prev_rank = rank_ - 1;
        int next_rank = rank_ + 1;

        // Exchange ALL necessary ghost cells with neighboring ranks.
        
        if(rank_ == 0){
        	// send up:
        	MPI_Send(&(rho_[0+local_N_*N_]), N_, MPI_DOUBLE, rank_+1, DATALINE_TAG, MPI_COMM_WORLD);
        	
        	// rcv from up:
        	MPI_Status status2;
		MPI_Recv(&(rho_[0+(local_N_+1)*N_]), N_, MPI_DOUBLE, rank_+1, DATALINE_TAG, MPI_COMM_WORLD, &status2);
        	
        }else if(rank_ == procs_-1){
        	// rcv from down:
        	MPI_Status status;
		MPI_Recv(&(rho_[0]), N_, MPI_DOUBLE, rank_-1, DATALINE_TAG, MPI_COMM_WORLD, &status);
        	
        	// send down:
        	MPI_Send(&(rho_[0+1*N_]), N_, MPI_DOUBLE, rank_-1, DATALINE_TAG, MPI_COMM_WORLD);
        }else{
        	// rcv from down:
        	MPI_Status status;
		MPI_Recv(&(rho_[0]), N_, MPI_DOUBLE, rank_-1, DATALINE_TAG, MPI_COMM_WORLD, &status);
        	
        	// send up:
        	MPI_Send(&(rho_[0+local_N_*N_]), N_, MPI_DOUBLE, rank_+1, DATALINE_TAG, MPI_COMM_WORLD);
        	
        	// rcv from up:
        	MPI_Status status2;
		MPI_Recv(&(rho_[0+(local_N_+1)*N_]), N_, MPI_DOUBLE, rank_+1, DATALINE_TAG, MPI_COMM_WORLD, &status2);
        	
        	// send down:
        	MPI_Send(&(rho_[0+1*N_]), N_, MPI_DOUBLE, rank_-1, DATALINE_TAG, MPI_COMM_WORLD);
        }
        
        
        /*
        if (prev_rank >= 0) {
            // TODO:MPI
            // ...
            
        } else {
            // TODO:MPI
            // ...
            
            
        }
	
	
	
        if (next_rank < procs_) {
            // TODO:MPI
            // ...
        } else {
            // TODO:MPI
            // ...
        }
        */

        // Central differences in space, forward Euler in time with Dirichlet
        // boundaries.
        for (int i = 2; i < local_N_; ++i) {
            for (int j = 1; j <= N_; ++j) {
                // TODO:DIFF
                // rho_tmp_[ ?? ] = ...
                
                // I can simply copy this from the omp code:
                // fac_ = dt_ * D_ / (dr_ * dr_)
                
                rho_tmp_[i * N_ + j] = rho_[i * N_ + j]
                                     + fac_ * (rho_[i * N_ + (j + 1)]
                                             + rho_[i * N_ + (j - 1)]
                                             + rho_[(i + 1) * N_ + j]
                                             + rho_[(i - 1) * N_ + j]
                                             - 4.0 * rho_[i * N_ + j]);
            }
        }

        // Update the first and the last rows of each rank.
        for (int i = 1; i <= local_N_; i += local_N_ - 1) {
            for (int j = 1; j <= N_; ++j) {
                // TODO:DIFF
                // rho_tmp_[ ?? ] = ...
                
                // I can simply copy this from the omp code:
                // fac_ = dt_ * D_ / (dr_ * dr_)
                
                rho_tmp_[i * N_ + j] = rho_[i * N_ + j]
                                     + fac_ * (rho_[i * N_ + (j + 1)]
                                             + rho_[i * N_ + (j - 1)]
                                             + rho_[(i + 1) * N_ + j]
                                             + rho_[(i - 1) * N_ + j]
                                             - 4.0 * rho_[i * N_ + j]);
            }
        }

        // Use swap instead of rho_ = rho_tmp__. This is much more efficient,
        // because it does not copy element by element, just replaces storage
        // pointers.
        using std::swap;
        swap(rho_tmp_, rho_);
    }

    void compute_diagnostics(const double t)
    {
        double heat = 0.0;

        // TODO:DIFF - Integration to compute heat
        // ...
        
        for (int i = 1; i <= local_N_; ++i) {
            for (int j = 1; j <= N_; ++j) {
     		heat += rho_tmp_[i * N_ + j];
            }
        }
        
	heat *= (dr_ * dr_);
	
        // TODO:MPI
        // ... --------------------------- implementing the reduction:
        
	if (rank_ == 0) {
		// only rcv
		double rcv_heat;
		MPI_Status status;
		MPI_Recv(&rcv_heat, 1, MPI_DOUBLE, rank_+1, DIAGNOSTICS_TAG, MPI_COMM_WORLD, &status);
		heat += rcv_heat;
	}else if(rank_ == (procs_-1)){
		// only send
		MPI_Send(&heat, 1, MPI_DOUBLE, rank_-1, DIAGNOSTICS_TAG, MPI_COMM_WORLD);
	}else{
		// rcv
		double rcv_heat;
		MPI_Status status;
		MPI_Recv(&rcv_heat, 1, MPI_DOUBLE, rank_+1, DIAGNOSTICS_TAG, MPI_COMM_WORLD, &status);
		heat += rcv_heat;
		// send
		MPI_Send(&heat, 1, MPI_DOUBLE, rank_-1, DIAGNOSTICS_TAG, MPI_COMM_WORLD);
	}
	
        if (rank_ == 0) {
//#if DEBUG
            //std::cout << "t = " << t << " heat = " << heat << '\n';
//#endif
            diag.push_back(Diagnostics(t, heat));
        }
    }

    void write_diagnostics(const std::string &filename) const
    {
        std::ofstream out_file(filename, std::ios::out);
        for (const Diagnostics &d : diag){
            out_file << d.time << '\t' << d.heat << '\n';
            //std::cout << d.time << '\t' << d.heat << '\n';
            }
        out_file.close();
    }

private:

    void initialize_density()
    {
        //TODO:DIFF Implement the initialization of the density distribution
        
        double bound = .5;
        
        int i_offset = (rank_-1)*N_/procs_;
        
        for (int i = 1; i <= local_N_; ++i) {
            for (int j = 1; j <= N_; ++j) {
                if (std::abs((i+i_offset - 1) * dr_ - .5 * L_) < bound
                        && std::abs((j - 1) * dr_ - .5 * L_) < bound) {
                    rho_[i * N_ + j] = 1;
                } else {
                    rho_[i * N_ + j] = 0;
                }
            }
        }
        
        
    }

    double D_, L_;
    int N_, Ntot_, local_N_, real_N_;
    double dr_, dt_, fac_;
    int rank_, procs_;

    std::vector<double> rho_, rho_tmp_;
    std::vector<Diagnostics> diag;
};


int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " D L N dt\n";
        return 1;
    }

    int rank, procs;
    //TODO:MPI Initialize MPI, number of ranks and number of processes involved in the communicator
    // ...
	MPI_Init(&argc, &argv);
	
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
 	MPI_Comm_size(MPI_COMM_WORLD,&procs);
	
	//std::cout << rank << ", " << procs << std::endl;

    const double D = std::stod(argv[1]);
    const double L = std::stod(argv[2]);
    const int N = std::stoul(argv[3]);
    const double dt = std::stod(argv[4]);

    Diffusion2D_MPI system(D, L, N, dt, rank, procs);

//#if DEBUG
    system.compute_diagnostics(0);
//#endif

    timer t;
    t.start();
    for (int step = 0; step < 10000; ++step) {
        system.advance();
//ifndef _PERF_
        system.compute_diagnostics(dt * step);
//#endif
    }
    t.stop();

    if (rank == 0)
        std::cout << "Timing: " << N << ' ' << t.get_timing() << '\n';

//#ifndef _PERF_
    if (rank == 0)
        system.write_diagnostics("diagnostics_mpi.dat");
//#endif

    // Finalize MPI
    // TODO:MPI
    // ...
     MPI_Finalize(); 

    return 0;
}
