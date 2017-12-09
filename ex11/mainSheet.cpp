#include <fstream>
#include <sstream>
#include <iomanip>
#include "common.h"
#include "ArrayOfParticles.h"
#include "VelocitySolverNSquared.h"

/*

problem size N^2 -> N = 1000 sqrt(p)

p=1
N=1000
#Ranks, Time - 1	20.8502	( 0.012568	20.3812 )

p=2
N=1414
#Ranks, Time - 2	21.08	( 0.037744	20.671 )

p=3
N=1732
#Ranks, Time - 3	21.2829	( 0.114791	20.8519 )

p=4
N=2000
#Ranks, Time - 4	25.6656	( 0.729635	24.5355 )

p=5
N=2236
#Ranks, Time - 5	45.5748	( 11.5315	33.1399 )

p=8
N=2828
#Ranks, Time - 8	53.2563	( 9.95763	41.9449 )

p=10
N=3162
#Ranks, Time - 10	78.5782	( 39.5503	37.4457 )

See plot in file:
x axis is p, y axis is time

*/

using namespace std;

void WriteTimeToFile(const int Nranks, const double time, const char * sFileName)
{
    ofstream outfile;
    outfile.open(sFileName, ios::out | ios::app);
    outfile << Nranks << "    " << scientific << setprecision(6) <<  time << "\n";
    outfile.close();
}

void Dump(ArrayOfParticles& dstParticles, ArrayOfParticles& allParticles, const int Np, const int NpProcess, const int step, const int rank)
{

    /* TODO 5: write particle position, velocity and circulation to file: only rank 0 should write the data */
    
    
    // not sure what the issue is with these gathers.
    // I could just implement sending and receiving in a loop, maybe even nonblocking comm.
    // but because of time I leave it here.
    
    //std::cout << "1" << std::endl;
    MPI_Gather(&(dstParticles.x[0]), NpProcess, MPI_DOUBLE,
    		&(allParticles.x[0]), NpProcess, MPI_DOUBLE,
    		0, MPI_COMM_WORLD);
    //std::cout << "2" << std::endl;
    MPI_Gather(&(dstParticles.y[0]), NpProcess, MPI_DOUBLE,
    		&(allParticles.y[0]), NpProcess, MPI_DOUBLE,
    		0, MPI_COMM_WORLD);
    //std::cout << "3" << std::endl;	
    MPI_Gather(&(dstParticles.u[0]), NpProcess, MPI_DOUBLE,
    		&(allParticles.u[0]), NpProcess, MPI_DOUBLE,
    		0, MPI_COMM_WORLD);
    		
    MPI_Gather(&(dstParticles.v[0]), NpProcess, MPI_DOUBLE,
    		&(allParticles.v[0]), NpProcess, MPI_DOUBLE,
    		0, MPI_COMM_WORLD);
    		
    MPI_Gather(&(dstParticles.gamma[0]), NpProcess, MPI_DOUBLE,
    		&(allParticles.gamma[0]), NpProcess, MPI_DOUBLE,
    		0, MPI_COMM_WORLD);
    
    
    if(rank == 0){
	    std::stringstream ss;
	    ss << "x, y, vx, vy, g\n";
	    
	    for (int i=0; i<allParticles.Np; i++)
	    {
	    	ss	<< dstParticles.x[i] << ", "
	    		<< dstParticles.y[i] << ", " 
	    		<< dstParticles.u[i] << ", " 
	    		<< dstParticles.v[i] << ", " 
	    		<< dstParticles.gamma[i] << "\n";
	    }
	    
	    
	    string content = ss.str();
	    MPI_Offset len = content.size();
	    char* data = const_cast<char *>(content.c_str());
	    
	    
	    MPI_File fh;
	    
	    std::stringstream fn;
	    fn << "dump_s_" << step << ".csv";
	    char * filename = const_cast<char *>(fn.str().c_str());
	    
	    MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
	    
	    MPI_File_set_size(fh, 0);
	    MPI_Offset base;
	    MPI_File_get_position(fh, &base);
	    MPI_Status status;
	    MPI_File_write_at(fh, base, data, len, MPI_CHAR, &status);
	    
	    MPI_File_close(&fh);
	    
    }
    
}

void DumpMPI(ArrayOfParticles& dstParticles, const int NpProcess, const int step, const int rank)
{

    /* TODO 5: write particle position, velocity and circulation to file: all ranks should write their data to a common file */
    
    // prepare data:
    std::stringstream ss;
    
    if(rank == 0){
    	ss << "x, y, vx, vy, g\n";
    }
    
    for (int i=0; i<dstParticles.Np; i++)
    {
    	ss	<< dstParticles.x[i] << ", "
    		<< dstParticles.y[i] << ", " 
    		<< dstParticles.u[i] << ", " 
    		<< dstParticles.v[i] << ", " 
    		<< dstParticles.gamma[i] << "\n";
    }
    
    string content = ss.str();
    MPI_Offset len = content.size();
    char* data = const_cast<char *>(content.c_str());
    
    //std::cout << "data: " << data << std::endl;
    
    // get offset:
    MPI_Offset offset = 0;
    MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    //std::cout << "offset: " << offset << std::endl;
    
    
    // go write data:
    MPI_File fh;
    
    std::stringstream fn;
    fn << "dump_" << step << ".csv";
    char * filename = const_cast<char *>(fn.str().c_str());
    
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    
    MPI_File_set_size(fh, 0);
    MPI_Offset base;
    MPI_File_get_position(fh, &base);
    MPI_Status status;
    MPI_File_write_at_all(fh, base + offset, data, len, MPI_CHAR, &status);
    
    MPI_File_close(&fh);
    
}

int main (int argc, char ** argv)
{
    int rank=0; // id of rank
    int size=1; // number of ranks
    /* TODO 1: initialization of MPI and setting of variables rank and size */
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // number of processes
    if (rank==0)
      std::cout << "Running with " << size << " MPI processes\n";
    
    // timer setup
    Timer timerIC, timerSim;
	
    // time integration setup
    const double dt = 0.001;
    const double tfinal = 2.5;
    const int ndump = 10;

    // output
    const bool bAnimation = true;
    const bool bVerbose = true;
    const bool dump_mpi = true;

    // number of particles
    const int N = 2236; // 10000
    size_t Np = (N/size)*size; // ensures that the number of particles is divisible by the number of workers
	
    timerIC.start();
    /* TODO 1: distribute particles such that each rank gets NpProcess particles */
    size_t NpProcess = Np/size;
    
    // particle vectors
    // dstParticles: particles owned by rank
    // srcParticles: particles with which the interaction has to be computed
    ArrayOfParticles dstParticles(NpProcess), srcParticles(NpProcess);
    ArrayOfParticles allParticles(rank==0 ? Np : 0); // list of all particles for output
    
    const double dx = 1./Np;
    double totGamma=0.; // total circulation is sum over gamma of all particles
    const double Gamma_s = 1.; 
    
    // initialize particles: position and circulation
    /* TODO 2: initialize particles (position and circulation) and compute total circulation totGamma */
    size_t i_offset = rank*NpProcess;
    double local_totGamma = 0.;
    
    for (size_t i=0; i<NpProcess; i++)
    {
    	double xx = -0.5 + (i+i_offset + 0.5)*dx;
	dstParticles.x[i] = xx;
	dstParticles.y[i] = 0;
	// -1/2 / sqrt( 1 - (x/0.5)^2 ) * -2 (x/0.5) * 1/0.5
	// = 4x / sqrt( 1 - 4 x^2)
	dstParticles.gamma[i] = dx * 4.0*xx/ sqrt(1.0 - 4*xx*xx);
	
	local_totGamma += dstParticles.gamma[i];
    }
    // allreduce totGamma:
    //std::cout << "[" << rank << "] " << local_totGamma << std::endl;
    MPI_Allreduce(&local_totGamma, &totGamma, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    timerIC.stop();

    if (rank==0)
    {
      std::cout << "Number of particles: " << Np << std::endl;
      std::cout << "Number of particles per process: " << NpProcess << std::endl;
      std::cout << "Initial circulation: " << totGamma << std::endl;
      std::cout << "IC time: " << timerIC.get_timing() << std::endl;
    }
    
    // initialize velocity solver
    VelocitySolverNSquared VelocitySolver(dstParticles, srcParticles, rank, size);
	
    timerSim.start();
    double t=0;
    int it=0;
    for (it=1; it<=std::ceil(tfinal/dt); it++)
    {
      // reset particle velocities
      dstParticles.ClearVelocities(); 
      // compute velocities corresponding to time n
      VelocitySolver.ComputeVelocity();
        
      // dump the particles
      if ((it-1)%ndump==0 && bAnimation)
      {
        if (!dump_mpi)
          Dump(dstParticles,allParticles,Np,NpProcess,it-1,rank);
        else
          DumpMPI(dstParticles,NpProcess,it-1,rank);
      }
        
      // update time
      if (rank==0)
        t += dt;
      /* TODO 1: communicate time to all other ranks */
        MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //std::cout << "[" << rank << ", " << it << "] " << t << std::endl;
        
      if (it%ndump==0 && rank==0 && bVerbose)
        std::cout << "Iteration " << it << " time " << t << std::endl;
		
      // advance particles in time to n+1 using forward Euler
      dstParticles.AdvectEuler(dt);
      // update "source" particles
      srcParticles = dstParticles;
    }
    // dump final state
    if ((it-1)%ndump==0 && bAnimation)
    {
      dstParticles.ClearVelocities();
      // compute velocities corresponding to time n
      VelocitySolver.ComputeVelocity();

      if (!dump_mpi)
        Dump(dstParticles,allParticles,Np,NpProcess,it-1,rank);
      else
        DumpMPI(dstParticles,NpProcess,it-1,rank);
    }
    
    if (bVerbose)
      std::cout << "Bye from rank " << rank << std::endl;
    timerSim.stop();
    
    if (rank==0)
    {
      char buf[500];
      sprintf(buf, "timing.dat");
      WriteTimeToFile(size,timerSim.get_timing(),buf);
      std::cout << "#Ranks, Time - " << size << "\t" << timerSim.get_timing() << "\t( " << VelocitySolver.timeT << "\t" << VelocitySolver.timeC << " )\n";
    }

    /* TODO 1: finalize MPI */
    MPI_Finalize();
    return 0;
}


