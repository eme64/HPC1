/*
 *
 *  Guido Novati: novatig@ethz.ch
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include "Utils.h"
//List of non-linearities for neural networks

struct Linear
{
  static inline void eval(nnOpInp in, nnOpRet out, const int N)
  {
    for (int i=0;i<N; i++) out[i] = in[i];
  }
  static inline void evalDiff(nnOpInp in, nnOpRet out, const int N)
  {
    //#pragma omp simd aligned(out, in : 32) safelen(64)
    //for (int i=0;i<N; i++) out[i] *= 1;
  }
  static Real weightsInitFactor(const int inps, const int outs)
  {
    return std::sqrt(6./(inps + outs));
  }
};

struct Tanh
{
  static inline void eval(nnOpInp in, nnOpRet out, const int N)
  {
    for (int i=0;i<N; i++) out[i] = eval(in[i]);
  }
  static inline void evalDiff(nnOpInp in, nnOpRet out, const int N)
  {
    for (int i=0;i<N; i++) out[i] *= evalDiff(in[i]);
  }
  static Real weightsInitFactor(const int inps, const int outs)
  {
    return std::sqrt(6./(inps + outs));
  }
  static inline Real eval(const Real in)
  {
    const Real e2x = std::exp(-2*in);
    return (1-e2x)/(1+e2x);
  }
  static inline Real evalDiff(const Real in)
  {
    const Real e2x = std::exp(-2.*in);
    return 4*e2x/((1+e2x)*(1+e2x));
  }
};

struct Relu
{
  static inline void eval(nnOpInp in, nnOpRet out, const int N)
  {
    for (int i=0;i<N; i++) out[i] = in[i]>0 ? in[i] : 0;
  }
  static inline void evalDiff(nnOpInp in, nnOpRet out, const int N)
  {
    for (int i=0;i<N; i++) out[i] *= in[i]>0 ? 1 : 0;
  }
  static Real weightsInitFactor(const int inps, const int outs)
  {
    return std::sqrt(2./inps);// 2./inps;
  }
};

#ifndef PRELU_FAC
#define PRELU_FAC 0.2
#endif
struct PRelu
{
  static inline void eval(nnOpInp in, nnOpRet out, const int N)
  {
    for (int i=0;i<N; i++) out[i] = in[i]>0 ? in[i] : PRELU_FAC*in[i];
  }
  static inline void evalDiff(nnOpInp in, nnOpRet out, const int N)
  {
    for (int i=0;i<N; i++) out[i] *= in[i]>0 ? 1 : PRELU_FAC;
  }
  static Real weightsInitFactor(const int inps, const int outs)
  {
    return std::sqrt(2./inps);// 2./inps;
  }
};

struct SoftMax
{
  static Real weightsInitFactor(const int inps, const int outs)
  {
    return std::sqrt(2./inps);
  }
  static inline void eval(nnOpRet in, nnOpRet out, const int N)
  {
    Real norm = nnEPS;
    for(int i=0; i<N; i++) {
      in[i] = std::exp(in[i]);
      norm += in[i];
    }
    for(int i=0; i<N; i++) out[i] = in[i]/norm;
  }
  static inline void evalDiff(nnOpInp in, nnOpRet out, const int N)
  {
    std::vector<Real> deltas(N);
    Real norm = nnEPS;
    for(int i=0; i<N; i++) {
      deltas[i] = out[i];
      norm += in[i];
      out[i] = 0;
    }

    for(int j=0; j<N; j++) {
        for(int i=0; i<N; i++)
          out[i] += deltas[j] * in[i] *((i==j) -in[j]/norm)/norm;
    }
  }
};
