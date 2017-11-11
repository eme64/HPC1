/*
 *
 *  Guido Novati: novatig@ethz.ch
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include <fstream>
#include "Network.h"

struct Adam {
  const Real eta, B1, B2;
  Adam(const Real _eta, const Real beta1, const Real beta2, const Real betat1, const Real betat2) :
  eta(_eta*std::sqrt(1-betat2)/(1-betat1)), B1(beta1), B2(beta2) {}

  inline Real step(const Real&grad, Real&M1, Real&M2)
  {
    M1 = B1 * M1 + (1-B1) * grad;
    M2 = B2 * M2 + (1-B2) * grad*grad;
    const Real _M2 = std::sqrt(M2 + nnEPS);
    return eta*M1/_M2;
  }
};

struct Momentum {
  const Real eta, B1;
  Momentum(const Real _eta, const Real beta1, const Real beta2,
    const Real betat1, const Real betat2) : eta(_eta), B1(beta1) {}

  inline Real step(const Real&grad, Real&M1, Real&M2)
  {
    M1 = B1 * M1 + eta * grad;
    return M1;
  }
};

struct SGD {
  const Real eta;
  SGD(const Real _eta, const Real beta1, const Real beta2,
    const Real betat1, const Real betat2) : eta(_eta) {}

  inline Real step(const Real&grad, Real&M1, Real&M2)
  {
    return eta*grad;
  }
};

template<typename Algorithm>
class Optimizer
{
protected:
  const Network& net;
  const Real eta, beta_1, beta_2;
  Real beta_t_1, beta_t_2;
  vector<Grads*> sumGrads, _1stMom, _2ndMom;

public:
  //Default parameters are valid for Adam! For SGD use 0.0001
  Optimizer(const Network& _net, const Real learn_rate = 0.001,
    const Real beta1 = 0.9, const Real beta2 = 0.999) : net(_net),
  eta(learn_rate),beta_1(beta1),beta_2(beta2),beta_t_1(beta1),beta_t_2(beta2)
  {
    sumGrads = net.allocateGrad();
    _1stMom = net.allocateGrad();
    _2ndMom = net.allocateGrad();
  }

  virtual ~Optimizer()
  {
    for(auto& p : sumGrads) _dispose_object(p);
    for(auto& p : _1stMom) _dispose_object(p);
    for(auto& p : _2ndMom) _dispose_object(p);
  }

  virtual void update(const int batchsize)
  {
    const vector<Params*>& P = net.params;
    const vector<vector<Grads*>>& G = net.thread_grads;
    const Real factor = 1./batchsize;
    Algorithm algo(eta, beta_1, beta_2, beta_t_1, beta_t_2);

    const size_t nThr = G.size(), nLayers = G[0].size();
    {
      //first, sum up gradients from each thread_grads
      for (size_t j=0; j<nLayers; j++) {
        if(G[0][j]==nullptr) {assert(!j); continue;} //input layer has no weights
        sumGrads[j]->clear(); //reset
        assert(G[0][j]->nWeights == sumGrads[j]->nWeights);
        assert(G[0][j]->nBiases  == sumGrads[j]->nBiases);
        assert(G[0][j]->nWeights == P[j]->nWeights);
        assert(G[0][j]->nBiases  == P[j]->nBiases);

        for (int w=0; w<G[0][j]->nWeights; w++)
          for (size_t t=0; t<nThr; t++)
            sumGrads[j]->weights[w] += factor*G[t][j]->weights[w];

        for (int w=0; w<G[0][j]->nBiases; w++)
          for (size_t t=0; t<nThr; t++)
            sumGrads[j]->biases[w] += factor*G[t][j]->biases[w];

        for (size_t t=0; t<nThr; t++) G[t][j]->clear();
      }

      //second, actually update the parameters with the averaged gradient:
      for (size_t j=0; j<nLayers; j++) {
        if(G[0][j] == nullptr) continue; //input layer

        for (int w=0; w<G[0][j]->nWeights; w++)
          P[j]->weights[w] += algo.step(sumGrads[j]->weights[w], _1stMom[j]->weights[w], _2ndMom[j]->weights[w]);
      }

      for (size_t j=0; j<nLayers; j++) {
        if(G[0][j] == nullptr) continue; //input layer

        for (int w=0; w<G[0][j]->nBiases; w++)
          P[j]->biases[w] += algo.step(sumGrads[j]->biases[w], _1stMom[j]->biases[w], _2ndMom[j]->biases[w]);
      }
    }

    // Needed by Adam optimization algorithm:
    beta_t_1 *= beta_1;
    if (beta_t_1<nnEPS) beta_t_1 = 0;
    beta_t_2 *= beta_2;
    if (beta_t_2<nnEPS) beta_t_2 = 0;
  }
};
