#include <vector>

#include "matrix.h"

namespace gaml {

namespace io {

arma::sp_fmat Matrix::fromStream(std::ifstream& f) {
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
}
}
