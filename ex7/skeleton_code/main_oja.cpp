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
#include <iostream>

// map from grayscale [0, 255] to Real [0, 1]
static void prepare_input(const std::vector<int>& image, std::vector<Real>& input)
{
  static const Real fac = 1/(Real)255;
  assert(image.size() == input.size());
  for (size_t j = 0; j < input.size(); j++) input[j] = image[j]*fac;
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
  
	
	int nthreads = 24;
	for(int i = 1; i < argc; i++ ) {
        	if( strcmp( argv[i], "-N" ) == 0 ) {
         	   nthreads = atoi(argv[i+1]);
         	   i++;
        	}
    	}
	
	omp_set_num_threads(nthreads);
	
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

  // Training parameters:
  const int nepoch = 10;
  const Real learn_rate = 1e-4;
  // Compression parameter:
  const int Z = 10;

  // Create Network:
  Network net;
  net.addInput<28*28*1>();
  net.addLayer<Linear, 28*28*1, Z>();
  net.initialize();

  // Random number generator to shuffle dataset:
  struct {
    std::mt19937 gen;
    inline size_t operator()(size_t n) {
      std::uniform_int_distribution<size_t> dist(0, n ? n-1 : 0);
      return dist(gen);
    }
  } generator;
	std::cout << "max threads: " << omp_get_max_threads() << std::endl;	
	std::cout << "num threads: " << omp_get_num_threads() << std::endl;
	std::cout << "start propagation and learning..." << std::endl;
	double ti1 = omp_get_wtime();
	
  for (int iepoch = 0; iepoch < nepoch; iepoch++)
  {
    std::vector<Real> input(28*28, 0);
    std::vector<int> sample_ids(n_train_samp);
    //fill array: 0, 1, ..., n_train_samp-1
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    std::random_shuffle(sample_ids.begin(), sample_ids.end(), generator);

    for (int sample = 0; sample < n_train_samp; sample++)
    {
      prepare_input(dataset.training_images[sample], input);
      net.predict(input);
      net.hebbian_learning(learn_rate);
    }
  }
	
	double ti2 = omp_get_wtime();
	std::cout << "finish with: " << ti2-ti1 << " sec" << std::endl;

  //extract features from the second layer (not implemented for the first layer)
  // WARNING: if you add layers to the net this will fail!
  for (int z = 0; z<Z; z++) {
    std::vector<float> component(28*28, 0);
    assert(net.params.back()->nWeights == 28*28*Z);
    for (int j = 0; j<28*28; j++) 
      component[j] = net.params.back()->weights[z + Z*j];

    FILE* pFile = fopen(("component_"+std::to_string(z)+".raw").c_str(),"wb");
    fwrite(component.data(), sizeof(float), 28*28, pFile);
    fflush(pFile); fclose(pFile);
  }

  return 0;
}
