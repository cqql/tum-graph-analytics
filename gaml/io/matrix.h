#ifndef GAML_IO_MATRIX_H_
#define GAML_IO_MATRIX_H_

#include <fstream>

#include <armadillo>

namespace gaml {

namespace io {

class Matrix {
 public:
  static arma::sp_fmat fromStream(std::ifstream& f);
};
}
}

#endif  // GAML_IO_MATRIX_H_
