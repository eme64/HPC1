/*
 *
 *  written by Guido Novati: novatig@ethz.ch
 *  Copyright 2017 ETH Zurich. All rights reserved.
 *
 */
#include "network/Network.h"
#include "network/Optimizer.h"
#include "mnist/mnist_reader.hpp"
#include <chrono>

// map from grayscale [0, 255] to Real [0, 1]
static void prepare_input(const std::vector<int>& image, std::vector<Real>& input)
{
  static const Real fac = 1/(Real)255;
  assert(image.size() == input.size());
  for (size_t j = 0; j < input.size(); j++) input[j] = image[j]*fac;
}

static Real compute_error(const std::vector<Real>& output, std::vector<Real>& input)
{
  Real l2err = 0;
  assert(output.size() == input.size());
  for (size_t j = 0; j < input.size(); j++) {
    l2err += std::pow(input[j] - output[j], 2);
    input[j] = input[j] - output[j];
  }
  return l2err / 2;
}

inline int test_dir()
{
  int missing_data = 0;
  FILE* dFile;
  dFile = fopen("../t10k-images-idx3-ubyte","rb");
  missing_data = missing_data || dFile == NULL;
  fclose(dFile);
  dFile = fopen("../t10k-labels-idx1-ubyte","rb");
  missing_data = missing_data || dFile == NULL;
  fclose(dFile);
  dFile = fopen("../train-images-idx3-ubyte","rb");
  missing_data = missing_data || dFile == NULL;
  fclose(dFile);
  dFile = fopen("../train-labels-idx1-ubyte","rb");
  missing_data = missing_data || dFile == NULL;
  fclose(dFile);
  return missing_data;
}

int main (int argc, char** argv)
{
  std::cout << "MNIST data directory: ../" << std::endl;
  if (test_dir()) {
    printf("Missing MNIST data, aborting... \n");
    abort();
  }
  // Load MNIST data"
  mnist::MNIST_dataset<std::vector, std::vector<int>, uint8_t> dataset =
  mnist::read_dataset<std::vector, std::vector, int, uint8_t>("../");
  assert(dataset.training_labels.size() == dataset.training_images.size());
  assert(dataset.test_labels.size() == dataset.test_images.size());
  const int n_train_samp = dataset.training_images.size();
  const int n_test_samp = dataset.test_images.size();

  // Training parameters:
  const int nepoch = 30, batchsize = 32;
  const Real learn_rate = 1e-4;
  // Compression parameter:
  const int Z = 10;

  // Create Network:
  Network net;
  net.addInput<28*28*1>();
  net.addLayer<Linear, 28*28*1, Z>();
  net.addLayer<Linear, Z, 28*28*1>();
  net.initialize();

  //Create optimizer:
  Optimizer<Adam> opt(net, learn_rate);

  #ifndef NDEBUG
  {
    Network net_test;
    net_test.addInput<28*28*1>();
    net_test.addLayer<Linear, 28*28*1, 1>();
    net_test.initialize();
    if (net_test.checkGrads()) {
      printf("Backpropagation failed the test.\n");
      abort();
    } else printf("Gradient check test passed.\n");
  }
  #endif

  // Random number generator to shuffle dataset:
  struct {
    std::mt19937 gen;
    inline size_t operator()(size_t n) {
      std::uniform_int_distribution<size_t> dist(0, n ? n-1 : 0);
      return dist(gen);
    }
  } generator;

  const int steps_in_epoch = n_train_samp / batchsize;
  assert(steps_in_epoch > 0);
  for (int iepoch = 0; iepoch < nepoch; iepoch++)
  {
    std::vector<int> sample_ids(n_train_samp);
    //fill array: 0, 1, ..., n_train_samp-1
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    std::random_shuffle(sample_ids.begin(), sample_ids.end(), generator);

    Real epoch_mse = 0;
    for (int step = 0; step < steps_in_epoch; step++)
    {
      std::vector<Real> input(28*28, 0);
    
      for (int i = 0; i < batchsize; i++)
      {
        const int sample = sample_ids[sample_ids.size()-1];
        prepare_input(dataset.training_images[sample], input);
        const std::vector<Real> prediction = net.predict(input);
        epoch_mse += compute_error(prediction,input); //now input contains err
        net.backProp(input);
        //remove indices already used for this batch:
        sample_ids.pop_back();
      }

      opt.update(batchsize);
    }

    if(iepoch % 10 == 0)
    {
      Real test_mse = 0;
      
      std::vector<Real> input(28*28, 0);
      
      for (int i = 0; i < n_test_samp; i++)
      {
        prepare_input(dataset.test_images[i], input);
        const std::vector<Real> prediction = net.predict(input);
        test_mse += compute_error(prediction,input); //now input contains err
      }
      printf("Training set MSE:%f, Test set MSE:%f\n",
        epoch_mse/n_train_samp, test_mse/n_test_samp);
    }
  }

  //extract features from the second layer (not implemented for the first layer)
  // WARNING: if you add layers to the net this will fail!
  for (int z = 0; z<Z; z++) {
    std::vector<float> component(28*28, 0);
    assert(net.params.back()->nWeights == 28*28*Z);
    for (int j = 0; j<28*28; j++)
      component[j] = net.params.back()->weights[j +28*28 * z];

    FILE* pFile = fopen(("component_"+std::to_string(z)+".raw").c_str(),"wb");
    fwrite(component.data(), sizeof(float), 28*28, pFile);
    fflush(pFile); fclose(pFile);
  }

  return 0;
}
