#include <fstream>

#include <armadillo>

#include "matrix_data.h"

// Parse the split matrix data into armadillo matrices
struct MatrixData MatrixData::parse(std::string path) {
  struct MatrixData md;
  std::ifstream f(path, std::ios::binary);

  if (!f.is_open()) {
    std::cout << path << std::endl;
    throw std::invalid_argument("Could not open file");
  }

  f.read(reinterpret_cast<char*>(&md.offset), sizeof(int));
  md.R = MatrixData::parsemat(f);

  return md;
}

arma::sp_fmat MatrixData::parsemat(std::ifstream& f) {
  int m;
  int n;
  int nnz;
  std::vector<unsigned int> rowdata;
  std::vector<unsigned int> coldata;
  std::vector<float> vals;

  // Read metadata
  f.read(reinterpret_cast<char*>(&m), sizeof(m));
  f.read(reinterpret_cast<char*>(&n), sizeof(n));
  f.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));

  // Prepare vectors, i.e. allocate memory
  rowdata.resize(nnz);
  coldata.resize(nnz);
  vals.resize(nnz);

  // Read matrix entries
  f.read(reinterpret_cast<char*>(rowdata.data()), nnz * sizeof(rowdata[0]));
  f.read(reinterpret_cast<char*>(coldata.data()), nnz * sizeof(coldata[0]));
  f.read(reinterpret_cast<char*>(vals.data()), nnz * sizeof(vals[0]));

  arma::urowvec rows(std::vector<arma::uword>(rowdata.begin(), rowdata.end()));
  arma::urowvec cols(std::vector<arma::uword>(coldata.begin(), coldata.end()));
  arma::umat locations = arma::join_cols(rows, cols);

  return arma::sp_fmat(locations, arma::fvec(vals), m, n);
}
