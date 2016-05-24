#ifndef MATRIX_DATA_H_
#define MATRIX_DATA_H_

#include <string>
#include <vector>

#include <armadillo>

// Parse the split matrix data into armadillo matrices
struct MatrixData {
  int offset;
  arma::sp_fmat R;

  static struct MatrixData parse(std::string path);

  static arma::sp_fmat parsemat(std::ifstream& f);
};

#endif  // MATRIX_DATA_H_
