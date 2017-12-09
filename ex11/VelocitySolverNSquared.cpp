
#include "VelocitySolverNSquared.h"

VelocitySolverNSquared::VelocitySolverNSquared(ArrayOfParticles & dstParticles, ArrayOfParticles & srcParticles, const int rank, const int size)
: dstParticles(dstParticles), srcParticles(srcParticles), rank(rank), size(size), timeC(0), timeT(0)
{
}

void VelocitySolverNSquared::ComputeVelocity()
{
    const double i2pi = 0.5/M_PI;
    const int N = dstParticles.Np;
    
    for (int pass=0; pass<size; pass++)
    {
        // 1. exchange
        Timer timerT;
        timerT.start();

        /* TODO 4: exchange particles */
	int rank_from = (rank - pass + size) % size;
	int rank_to = (rank + pass) % size;
	
	// need to communicate x,y,gamma (u,v only needed locally!)
	MPI_Request req[6];
	MPI_Status status[6];
	MPI_Irecv(&(srcParticles.x[0]), N, MPI_DOUBLE, rank_from, 8001, MPI_COMM_WORLD, &req[0]);
	MPI_Irecv(&(srcParticles.y[0]), N, MPI_DOUBLE, rank_from, 8002, MPI_COMM_WORLD, &req[1]);
	MPI_Irecv(&(srcParticles.gamma[0]), N, MPI_DOUBLE, rank_from, 8003, MPI_COMM_WORLD, &req[2]);
	
	MPI_Isend(&(dstParticles.x[0]), N, MPI_DOUBLE, rank_to, 8001, MPI_COMM_WORLD, &req[3]);
	MPI_Isend(&(dstParticles.y[0]), N, MPI_DOUBLE, rank_to, 8002, MPI_COMM_WORLD, &req[4]);
	MPI_Isend(&(dstParticles.gamma[0]), N, MPI_DOUBLE, rank_to, 8003, MPI_COMM_WORLD, &req[5]);
	
	MPI_Waitall(6, req, status);
	
        timerT.stop();
        timeT += timerT.get_timing();
        
        // 2. compute local
        Timer timerC;
        timerC.start();

        /* TODO 4: compute induced velocities by received particles at owned particle positions */
	for (int my_i=0; my_i<N; my_i++)
        {
        	double my_x = dstParticles.x[my_i];
		double my_y = dstParticles.y[my_i];
		double my_gamma = dstParticles.gamma[my_i];
		
        	for (int rcv_i=0; rcv_i<N; rcv_i++)
		{
			if(rank_from != rank || my_i != rcv_i){
				double rcv_x = srcParticles.x[rcv_i];
				double rcv_y = srcParticles.y[rcv_i];
				double rcv_gamma = srcParticles.gamma[rcv_i];
			
				double squareDist = (my_x - rcv_x)*(my_x - rcv_x) + (my_y - rcv_y)*(my_y - rcv_y);
			
				dstParticles.u[my_i] += i2pi* rcv_gamma * (rcv_y - my_y) / squareDist;
				dstParticles.v[my_i] += i2pi* rcv_gamma * (my_x - rcv_x) / squareDist;
			}
		}
        }
	
        timerC.stop();
        timeC += timerC.get_timing();
    }
}
