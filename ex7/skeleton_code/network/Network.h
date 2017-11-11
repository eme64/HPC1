/*
 *
 *  Guido Novati: novatig@ethz.ch
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layer_Normal.h"

class Network
{
public:
  std::random_device rd;
  std::mt19937 gen;
  vector<Params*> params;
  vector<Layer*> layers;
  vector<vector<Grads*>> thread_grads;
  vector<vector<Activation*>> thread_activations;
  int nInputs=0, nOutputs=0;

  // Network() : gen(0) {};
  Network() : gen(rd()) {};

  ~Network()
  {
    for(auto& p : params) _dispose_object(p);
    for(auto& p : layers) _dispose_object(p);
    for(auto& a : thread_grads)       for(auto& p : a) _dispose_object(p);
    for(auto& a : thread_activations) for(auto& p : a) _dispose_object(p);
  }

  vector<Real> predict(const vector<Real>& inp) const
  {
    const size_t nLayers = layers.size();
    if(thread_activations.size()==0 || thread_grads.size()==0) {
      printf("Attempted to access uninitialized network. Aborting\n");
      abort();
    }
    //if you are working on oja's rule, pretend you do not see omp stuff here
    // with hebbian learning multithreading must occur inside predict
    const int thrID = omp_get_thread_num();
    //get a thread-safe memory space to compute predition
    const vector<Activation*>& act = thread_activations[thrID];
    // clean up memory space from previous prediction:
    for(auto& a : act) { a->clearInputs(); a->clearOutput(); }
    assert(nInputs==(int)inp.size() && nInputs==act.front()->size);

    //copy input onto output of input layer:
    for (int j=0; j<nInputs; j++) act.front()->outvals[j] = inp[j];

    for (size_t j=0; j<nLayers; j++) layers[j]->propagate(act, params);

    vector<Real> ret(nOutputs);
    assert(nOutputs==act.back()->size);
    //copy output of output layer onto return vector:
    for (int i=0; i<nOutputs; i++) ret[i] = act.back()->outvals[i];
    return ret;
  }

  void hebbian_learning(const Real beta) const
  {
    const size_t nLayers = layers.size();
    const int thrID = omp_get_thread_num();
    assert(thrID == 0); // does not make sense to parallelize at this level
    //get a thread-safe memory space to compute predition
    const vector<Activation*>& act = thread_activations[thrID];

    for(size_t j=0; j<nLayers; j++) layers[j]->ojaUpdate(act, params, beta);
  }

  void backProp(const vector<Real>& err) const
  {
    //this function assumes that the same thread already performed prediction
    //and computed some error given input, this only computes gradient
    const size_t nLayers = layers.size();
    if(thread_activations.size()==0 || thread_grads.size()==0) {
      printf("Attempted to access uninitialized network. Aborting\n");
      abort();
    }
    const int thrID = omp_get_thread_num();
    const vector<Grads*>& G = thread_grads[thrID];
    const vector<Activation*>& act = thread_activations[thrID];
    for(auto& a : act) a->clearErrors();

    assert(nOutputs==(int)err.size() && nOutputs==act.back()->size);
    for (int i=0; i<nOutputs; i++) act.back()->errvals[i] = err[i];

    //backprop starts at the last layer:
    for (size_t i=nLayers; i>0; i--) layers[i-1]->backPropagate(act, params, G);
  }

  void initialize()
  {
    // allocate work memory: space for gradients and network compute
    // thread safety: allocate for each thread
    const int nThr = omp_get_max_threads();
    thread_activations.resize(nThr);
    thread_grads.resize(nThr);
    for(int i=0; i<nThr; i++) {
      thread_grads[i] = allocateGrad();
      thread_activations[i] = allocateActivation();
    }
  }

  void save() const
  {
    for(const auto &l : layers) l->save(params);
  }
  void restart() const
  {
    for(const auto &l : layers) l->restart(params);
  }

  template<int size>
  void addInput()
  {
    if(size<=0) { printf("Requested empty layer. Aborting.\n"); abort(); }
    if(layers.size() not_eq 0) {
      printf("Multiple input layers. Aborting.\n");
      abort();
    }

    Layer * l = new Input_Layer(size);
    nInputs = size;
    layers.push_back(l);
    params.push_back(nullptr);
  }

  template<typename func, int nInputs, int nNeurons>
  void addLayer(const std::string fname = std::string())
  {
    if(!layers.size()) { printf("Missing input layer. Aborting.\n"); abort(); }
    if(nInputs<=0 || nNeurons<= 0) {
      printf("Requested empty layer. Aborting.\n");
      abort();
    }
    if(layers.back()->nOutputs() not_eq nInputs) {
      printf("Mismatch between input size (%d) and previous layer size (%d). Aborting\n", nInputs, layers.back()->nOutputs());
      abort();
    }
    auto l = new NormalLayer<func>(nInputs, nNeurons, layers.size());

    layers.push_back(l);
    nOutputs = l->nOutputs();
    params.push_back(l->allocate_params());

    if(fname not_eq std::string())
      params.back()->restart(fname);
    else l->initialize(&gen, params);
  }

  inline vector<Activation*> allocateActivation() const
  {
    vector<Activation*> ret(layers.size(), nullptr);
    for(size_t j=0; j<layers.size(); j++)
      ret[j] = layers[j]->allocateActivation();
    return ret;
  }
  inline vector<Grads*> allocateGrad() const
  {
    vector<Grads*> ret(params.size(), nullptr);
    for(size_t j=0; j<params.size(); j++)
      if(params[j] not_eq nullptr)
        ret[j] = params[j]->allocateGradient();
    return ret;
  }

  int checkGrads()
  {
    printf("Checking gradients\n");
    assert(thread_grads.size() > 1);
    const Real incr = std::sqrt(2.2e-16), tol  = std::sqrt(incr);
    for (int o=0; o<nOutputs; o++)
    {
      vector<Real> res(nOutputs), inputs(nInputs), errs(nOutputs);
      const vector<Grads*>& grads = thread_grads[0];
      errs[0] = -1;
      for(auto& a: thread_activations) for(auto& p: a) p->clearErrors();

      normal_distribution<Real> dis_inp(0, 2);
      for(int j=0; j<nInputs; j++) inputs[j]= dis_inp(gen);

      predict(inputs);
      backProp(errs);

      Real meanerr = 0, squarederr = 0, cnterr = 0;
      for (size_t j=1; j<layers.size(); j++)
      for (int i=0; i < params[j]->nWeights + params[j]->nBiases; i++)
      {
        int w = i < params[j]->nWeights ? i : i - params[j]->nWeights;
        const Real grad= i<params[j]->nWeights? grads[j]->weights[w] : grads[j]->biases[w];

        //1
        if(i < params[j]->nWeights) params[j]->weights[w] += incr;
        else  params[j]->biases[w] += incr;
        res = predict(inputs);
        Real diff = -res[o]/(2*incr);

        //2
        if(i < params[j]->nWeights) params[j]->weights[w] -= 2*incr;
        else  params[j]->biases[w] -= 2*incr;
        res = predict(inputs);
        diff += res[o]/(2*incr);

        //0
        if(i < params[j]->nWeights) params[j]->weights[w] += incr;
        else  params[j]->biases[w] += incr;

        const Real scale = std::max(std::fabs(grad), std::fabs(diff));
        const Real err = std::fabs(grad-diff);
        if (scale < 2.2e-16) printf("Scale is zero, continuing\n");
        else if (err > tol) printf("Absolute error is %f\n", err);
        //if(i < params[j]->nWeights) printf("%f\n",params[j]->weights[w]);
        //else                        printf("%f\n",params[j]->biases[w]);

        cnterr += 1;
        meanerr += err;
        squarederr += err*err;
      }

      const Real stdef= std::sqrt((squarederr -meanerr*meanerr/cnterr)/cnterr);
      const Real mean = meanerr/cnterr;
      printf("Mean err:%g (std:%g).\n", mean, stdef);
      if(mean > 1e-7) return 1;
    }
    return 0;
  }
};
