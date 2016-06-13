#ifndef GAML_IO_MATRIX_SLICE_H_
#define GAML_IO_MATRIX_SLICE_H_

#include <string>
#include <vector>

#include <armadillo>

namespace gaml {

namespace io {

struct MatrixSlice {
  int offset;
  arma::sp_fmat R;

  static struct MatrixSlice parse(std::string path);
};
}
}

#endif  // GAML_IO_MATRIX_SLICE_H_
