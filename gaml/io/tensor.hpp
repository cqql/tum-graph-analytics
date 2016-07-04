#ifndef GAML_IO_TENSOR_HPP_
#define GAML_IO_TENSOR_HPP_

#include <string>
#include <vector>

#include <armadillo>

namespace gaml {
namespace io {

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
  
  arma::frowvec getWordBagAt(unsigned int i) const;
};


// Parse the split tensor data
struct TensorSlice {
  int offset;
  Sparse3dTensor R;
  
  static struct TensorSlice parse(std::string path);
  static struct Sparse3dTensor parseTensor(std::ifstream& f);
};
} // end io
} // end gaml

#endif  // GAML_IO_TENSOR_HPP_


