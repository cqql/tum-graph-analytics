#ifndef MATRIX_DATA_H_
#define MATRIX_DATA_H_

#include <string>
#include <vector>

#include <armadillo>


struct Sparse3dTensor {
  
  // size in each dim
  unsigned int n_rows;
  unsigned int n_cols;
  unsigned int n_words;
  
  // number of non-zero word bags
  unsigned int n_nz;
  // number of non-zero word bag elements
  unsigned int n_vals;
  
  // indices
  std::vector<unsigned int> rows;
  std::vector<unsigned int> cols;
  std::vector<unsigned int> bags;
  std::vector<unsigned int> words;
  std::vector<float> vals;
  
  arma::frowvec getWordBagAt(unsigned int i);
};


// Parse the split tensor data
struct TensorData {
  int offset;
  Sparse3dTensor R;
  
  static struct TensorData parse(std::string path);
  static struct Sparse3dTensor parseTensor(std::ifstream& f);
};


#endif  // MATRIX_DATA_H_

