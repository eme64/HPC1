/*
 *
 *  Guido Novati: novatig@ethz.ch
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include <random>
#include <vector>
#include <array>
#include <cassert>
#include <sstream>
#include <cstring>
#include <utility>
#include <limits>
#include <cmath>
#include <algorithm>
#include <immintrin.h>

#include <omp.h>

using namespace std;

#if 1
  typedef double Real;
#else
  typedef float Real;
#endif

typedef Real* __restrict__       const nnOpRet;
typedef const Real* __restrict__ const nnOpInp;

static const int simdWidth = 32/sizeof(Real);
static const Real nnEPS = std::numeric_limits<Real>::epsilon();

inline int roundUpSimd(const int size)
{
  return std::ceil(size/(Real)simdWidth)*simdWidth;
}

static inline Real readCutStart(vector<Real>& buf)
{
  const Real ret = buf.front();
  buf.erase(buf.begin(),buf.begin()+1);
  assert(!std::isnan(ret) && !std::isinf(ret));
  return ret;
}
static inline Real readBuf(vector<Real>& buf)
{
  //const Real ret = buf.front();
  //buf.erase(buf.begin(),buf.begin()+1);
  const Real ret = buf.back();
  buf.pop_back();
  assert(!std::isnan(ret) && !std::isinf(ret));
  return ret;
}
static inline void writeBuf(const Real weight, vector<Real>& buf)
{
  buf.insert(buf.begin(), weight);
}

template <typename T>
inline void _myfree(T *const& ptr)
{
  if(ptr == nullptr) return;
  free(ptr);
}

inline void _allocate(Real*& ptr, const int size)
{
  const int sizeSIMD = roundUpSimd(size)*sizeof(Real);
  posix_memalign((void **) &ptr, 32, sizeSIMD);
  memset(ptr, 0, sizeSIMD);
}

inline Real* init(const int N)
{
  Real* ret;
  _allocate(ret, N);
  return ret;
}

template <typename T>
void _dispose_object(T *& ptr)
{
    if(ptr == nullptr) return;
    delete ptr;
    ptr=nullptr;
}

template <typename T>
void _dispose_object(T *const& ptr)
{
    if(ptr == nullptr) return;
    delete ptr;
}
