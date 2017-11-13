/*
 *
 *  Guido Novati: novatig@ethz.ch
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"
#include <iostream>

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
	// computing output of neurons:
	//std::cout << "doing prediction..." << std::endl;
	#pragma omp parallel for
	
		for(int j=0; j<nNeurons; j++){
			// over all neurons
			suminp[j] = 0;
			for(int k=0; k<nInputs; k++){
				// not sure about weight layout, now more sure...
				suminp[j]+= weight[j + nNeurons*k] * inputs[k];
			}
			
		}
	//std::cout << "done." << std::endl;

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
	// w(j, i+1) = w(j,i) + beta * y(j,i) ( x(i) - w_j(i) * y(j,i) - 2* sum( y(k, i) * w(k,i) ) )
	// vec |        vec |   scal   scal     vec|   vec|    scal     vec |   scal      vec |
	
	// problem will be that some want to access old weights that are already overwritten -> first make old_copy:
	Real old_weight [nNeurons * nInputs];
	#pragma omp parallel for collapse(2)
	for(int i=0;i<nInputs; i++){
		for(int k=0; k<nNeurons;k++){
			old_weight[k + i*nNeurons] = weight[k + i+nNeurons];
		}
	}
	
	// guided schedule because for bigger j the sum over k will be bigger
	#pragma omp parallel for collapse(2) schedule(guided) 
	for(int j=0; j<nNeurons;j++){// reverse order to not destroy lower j's
		for(int i=0;i<nInputs;i++){// all entries of vector
			Real locsum = 0;
			
			for(int k=0; k<j; k++){
				locsum += output[k] * old_weight[k + i*nNeurons];
			}
			
			weight[j + i*nNeurons] = old_weight[j + i*nNeurons] + learnRate * output[j] * (
				inputs[i]
				-old_weight[j + i*nNeurons] * output[j]
				-2.0* locsum
			);
		}
	}

	/* ---------- sequential code.
	for(int j=nNeurons-1; j>=0;j--){// reverse order to not destroy lower j's
                for(int i=0;i<nInputs;i++){// all entries of vector
                        Real locsum = 0;

                        for(int k=0; k<j; k++){
                                locsum += output[k] * weight[k + i*nNeurons];
                        }

                        weight[j + i*nNeurons] += learnRate * output[j] * (
                                inputs[i]
                                -weight[j + i*nNeurons] * output[j]
                                -2.0* locsum
                        );
                }
        }
	*/
	//std::cout << "done." << std::endl;
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
