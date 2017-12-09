
#ifndef VELOCITYSOLVERNSQUARED_H
#define VELOCITYSOLVERNSQUARED_H

#include "common.h"
#include "ArrayOfParticles.h"

class VelocitySolverNSquared
{
private:
    ArrayOfParticles & dstParticles;
    ArrayOfParticles & srcParticles;
    
    const int rank, size;
    
public:
    VelocitySolverNSquared(ArrayOfParticles & dstParticles, ArrayOfParticles & srcParticles,const int rank,const int size);
    ~VelocitySolverNSquared(){}
    
    void ComputeVelocity();
    
    double timeC, timeT;
};

#endif /* VELOCITYSOLVERNSQUARED_H */
