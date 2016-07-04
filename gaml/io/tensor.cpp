#include <fstream>

#include <armadillo>

#include "tensor.hpp"

namespace gaml {
namespace io {
// parse split 
struct TensorSlice TensorSlice::parse(std::string path) {
  TensorSlice ts;
  std::ifstream f(path, std::ios::binary);
  
  if (!f.is_open()) {
    std::cout << path << std::endl;
    throw std::invalid_argument("Could not open file");
  }
  
  f.read(reinterpret_cast<char*>(&ts.offset), sizeof(offset));
  ts.R = parseTensor(f);
  return ts;
}

arma::frowvec Sparse3dTensor::getWordBagAt(unsigned int i) const{
  arma::frowvec wordbag(n_words, arma::fill::zeros);
  std::size_t start = 0;
  std::size_t end = bags[i];
  if(i != 0){
    start = bags[i-1];
  }
  
  for(auto d = start; d != end; ++d){
    wordbag(words[d]) = vals[d];
  }
  return wordbag;
}

struct Sparse3dTensor TensorSlice::parseTensor(std::ifstream& f) {
  Sparse3dTensor R;
  // Read metadata
  f.read(reinterpret_cast<char*>(&(R.n_rows)), sizeof(R.n_rows));
  f.read(reinterpret_cast<char*>(&(R.n_cols)), sizeof(R.n_cols));
  f.read(reinterpret_cast<char*>(&(R.n_words)), sizeof(R.n_words));
  f.read(reinterpret_cast<char*>(&(R.n_nz)), sizeof(R.n_nz));
  f.read(reinterpret_cast<char*>(&(R.n_vals)), sizeof(R.n_vals));
  
  // Prepare vectors, i.e. allocate memory
  R.rows.resize(R.n_nz);
  R.cols.resize(R.n_nz);
  R.bags.resize(R.n_nz);
  R.words.resize(R.n_vals);
  R.vals.resize(R.n_vals);
  
  // Read tensor entries
  f.read(reinterpret_cast<char*>(R.rows.data()), R.n_nz * sizeof(R.rows[0]));
  f.read(reinterpret_cast<char*>(R.cols.data()), R.n_nz * sizeof(R.cols[0]));
  f.read(reinterpret_cast<char*>(R.bags.data()), R.n_nz * sizeof(R.bags[0]));
  f.read(reinterpret_cast<char*>(R.words.data()), R.n_vals * sizeof(R.words[0]));
  f.read(reinterpret_cast<char*>(R.vals.data()), R.n_vals * sizeof(R.vals[0]));
  return R;
}
}// end io
}// end gaml

