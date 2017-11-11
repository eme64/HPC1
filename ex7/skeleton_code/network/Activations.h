/*
 *
 *  Guido Novati: novatig@ethz.ch
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include "Functions.h"

struct Activation
{
  const int size;
  Activation(const int _size) : size(_size),
    //contains all inputs to each neuron (inputs to network input layer is empty)
    suminps(init(_size)),
    //contains all neuron outputs that will be the incoming signal to linked layers (outputs of input layer is network inputs)
    outvals(init(_size)),
    //deltas for each neuron
    errvals(init(_size)) { assert(_size>0); }
  ~Activation() { _myfree(suminps); _myfree(outvals); _myfree(errvals); }
  inline void clearOutput() { std::memset(outvals,0,size*sizeof(Real)); }
  inline void clearErrors() { std::memset(errvals,0,size*sizeof(Real)); }
  inline void clearInputs() { std::memset(suminps,0,size*sizeof(Real)); }
  Real*const suminps;
  Real*const outvals;
  Real*const errvals;
};

struct Grads
{
  const int nWeights, nBiases;

  Grads(const int _nWeights, const int _nBiases): nWeights(_nWeights), nBiases(_nBiases), weights(init(_nWeights)), biases(init(_nBiases)) { }

  ~Grads()  { _myfree(weights); _myfree(biases); }
  inline void clear()
  {
    std::memset(weights, 0, nWeights*sizeof(Real));
    std::memset(biases, 0, nBiases*sizeof(Real));
  }
  Real*const weights;
  Real*const biases;
};

struct Params
{
  const int nWeights, nBiases;
  Params(const int _nWeights, const int _nBiases): nWeights(_nWeights), nBiases(_nBiases), weights(init(_nWeights)), biases(init(_nBiases)) { }

  ~Params() { _myfree(weights); _myfree(biases); }
  Real* const weights;
  Real* const biases;

  Grads* allocateGradient()
  {
    return new Grads(nWeights, nBiases);
  }

  void save(const std::string fname) const
  {
    FILE* wFile=fopen(("W_"+fname+".raw").c_str(),"wb");
    FILE* bFile=fopen(("b_"+fname+".raw").c_str(),"wb");
    fwrite(weights, sizeof(Real), nWeights, wFile);
    fwrite(biases,  sizeof(Real),  nBiases, bFile);
    fflush(wFile); fflush(bFile);
    fclose(wFile); fclose(bFile);
  }

  void restart(const std::string fname)
  {
    FILE* wFile=fopen(("W_"+fname+".raw").c_str(),"rb");
    FILE* bFile=fopen(("b_"+fname+".raw").c_str(),"rb");

    size_t wsize = fread(weights, sizeof(Real), nWeights, wFile);
    fclose(wFile);
    if((int)wsize not_eq nWeights){
      printf("Mismatch in restarted weight file %s; container:%lu read:%d. Aborting.\n", fname.c_str(), wsize, nWeights);
      abort();
    }

    size_t bsize = fread(biases, sizeof(Real),  nBiases, bFile);
    fclose(bFile);
    if((int)bsize not_eq nBiases){
      printf("Mismatch in restarted biases file %s; container:%lu read:%d. Aborting.\n", fname.c_str(), bsize, nBiases);
      abort();
    }
  }
};
