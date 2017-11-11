/*
 *
 *  Guido Novati: novatig@ethz.ch
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Activations.h"

class Layer
{
 public:
  const int size, ID;
  inline int nOutputs() const { return size; }
  Layer(const int _size, const int _ID) : size(_size), ID(_ID) {}
  virtual ~Layer() {}

  virtual void propagate(const vector<Activation*>& act, const vector<Params*>& param) const = 0;
  virtual void ojaUpdate(const vector<Activation*>& act, const vector<Params*>& param, const Real learnRate) const = 0;
  virtual void backPropagate(const vector<Activation*>& act, const vector<Params*>& param, const vector<Grads*>& grad) const = 0;

  virtual void initialize(mt19937*const gen, const vector<Params*>& param) const = 0;

  virtual void save(const vector<Params*>& param) const
  {
    if(param[ID] == nullptr) { assert(ID==0); return; } //input layer
    //param[ID]->save(ID);
  };

  virtual void restart(const vector<Params*>& param) const
  {
    if(param[ID] == nullptr) { assert(ID==0); return; } //input layer
    //param[ID]->restart(ID);
  };

  Activation* allocateActivation()
  {
    return new Activation(nOutputs());
  }
};

class Input_Layer: public Layer
{
 public:
  Input_Layer(const int _size) : Layer(_size, 0) {}

  void propagate(const vector<Activation*>& act, const vector<Params*>& param) const override {}

  void ojaUpdate(const vector<Activation*>& act, const vector<Params*>& param, const Real learnRate) const override {}
  
  void backPropagate(const vector<Activation*>& act, const vector<Params*>& param, const vector<Grads*>& grad) const override {}

  void initialize(mt19937*const gen, const vector<Params*>& param) const override {}
};
