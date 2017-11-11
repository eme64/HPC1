/*
 *
 *  Guido Novati: novatig@ethz.ch
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"

template<typename func>
class NormalLayer: public Layer
{
 public:
  const int nInputs, nNeurons;
  inline Params* allocate_params()
  {
    return new Params(nInputs*nNeurons, nNeurons);
  }
  NormalLayer(const int _nInputs, const int _nNeurons, const int _ID) : Layer(_nNeurons, _ID), nInputs(_nInputs), nNeurons(_nNeurons)
  {
    printf("\nNormal Layer of sizes:\nInput:%d Output:%d\n\n",
      nInputs, nNeurons);
    fflush(0);
      assert(nNeurons>0 && nInputs>0);
  }

  // Compute h = W x + b, y = func(h)
  // act: contains outputs of previous layers during current forward propagate
  // param: contains weight matrices and bias vectors
  void propagate(const vector<Activation*>& act, const vector<Params*>& param) const override
  {
    //array of outputs from previous layer:
    const Real* const inputs = act[ID-1]->outvals; //size is nInputs
    //weight matrix and bias vector:
    const Real* const weight = param[ID]->weights; //size is nNeurons * nInputs
    const Real* const bias   = param[ID]->biases; //size is nNeurons
    //return array that contains weight * input (matrix vector mul):
          Real* const suminp = act[ID]->suminps; //size is nNeurons
    //return array that contains func(weight * input):
          Real* const output = act[ID]->outvals; //size is nNeurons

    //////// TODO: Implement prediction:


    //////// TODO

    //apply function to suminp array. for Oja's rule assume this is linear
    //and therefore func copies suminp onto output
    func::eval(suminp, output, nNeurons);
  }

  // modify weight and bias using oja's update rule
  void ojaUpdate(const vector<Activation*>& act, const vector<Params*>& param, const Real learnRate) const override
  {
    //same comments as in function propagate apply to these arrays:
    const Real* const inputs = act[ID-1]->outvals;
    const Real* const output = act[ID]->outvals;
          Real* const weight = param[ID]->weights;
          Real* const bias   = param[ID]->biases;

    // Skeleton as is assumes that predict has been run before
    // therefore array output already contains the Y's predicted by the net

    //////// TODO: Implement Oja's Rule:


    //////// TODO
  }

  void backPropagate(const vector<Activation*>& act, const vector<Params*>& param, const vector<Grads*>& grad) const override
  {
    // act[ID]->errvals already contains dError/d output of a layer
    func::evalDiff(act[ID]->suminps, act[ID]->errvals, size);
    // After this ^ function, act[ID]->errvals contains dError/d suminp
    // (again remembering notation that output = func(suminp) )
    const Real* const deltas = act[ID]->errvals;   //size is nNeurons
    const Real* const inputs = act[ID-1]->outvals; //size is nInputs
    const Real* const weight = param[ID]->weights; //size is nInputs*nNeurons

    //These arrays are to be filled by backprop:
    //  errinp will contain dError/d output of the previous layer
          Real* const errinp = act[ID-1]->errvals; //size is nInputs
    //  these contain dError / d Parameter:
          Real* const gradW  = grad[ID]->weights;  //size is nInputs*nNeurons
          Real* const gradB  = grad[ID]->biases;   //size is nNeurons

    // For example, bias gradient is:
    // (plus equal because we take the update by averaging over some gradients)
    for(int n=0; n<nNeurons; n++)
      gradB[n] += deltas[n];

    //////// TODO: Implement BackProp to compute gradW and errinp


    //////// TODO
  }

  void initialize(mt19937* const gen, const vector<Params*>& param) const override
  {
    const Real scale = func::weightsInitFactor(nInputs, nNeurons);
    uniform_real_distribution<Real> dis(-scale,scale);
    assert(param[ID]->nWeights == nInputs*nNeurons);
    assert(param[ID]->nBiases == nNeurons);

    for (int o = 0; o < nNeurons; o++) {
      param[ID]->biases[o] = dis(*gen);
      for (int i = 0; i < nInputs; i++)
          param[ID]->weights[o +nNeurons*i] = dis(*gen);
    }
  }
};
