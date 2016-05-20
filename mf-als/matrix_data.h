#ifndef MATRIX_DATA_H_
#define MATRIX_DATA_H_

#include <string>
#include <vector>

#include <armadillo>

// Parse the split matrix data into armadillo matrices
struct MatrixData {
  int prodoffset;
  int useroffset;
  arma::sp_fmat Rprod;
  arma::sp_fmat Ruser;

  static struct MatrixData parse(std::string path);

  static arma::sp_fmat parsemat(std::ifstream& f);
};

#endif  // MATRIX_DATA_H_
