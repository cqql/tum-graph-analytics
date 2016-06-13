#include <fstream>

#include "matrix_slice.h"
#include "matrix.h"

namespace gaml {

namespace io {

// Parse the matrix slice into an armadillo matrix
struct MatrixSlice MatrixSlice::parse(std::string path) {
  struct MatrixSlice ms;
  std::ifstream f(path, std::ios::binary);

  if (!f.is_open()) {
    std::cout << path << std::endl;
    throw std::invalid_argument("Could not open file");
  }

  f.read(reinterpret_cast<char*>(&ms.offset), sizeof(int));
  ms.R = Matrix::fromStream(f);

  return ms;
}
}
}
