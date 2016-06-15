#ifndef GAML_MF_GD_PROJECTION_H_
#define GAML_MF_GD_PROJECTION_H_

#include <armadillo>

namespace gaml {

namespace mf {

namespace gd {

class Projection {
 public:
  virtual arma::fmat project(arma::fmat) const = 0;
};
}
}
}

#endif  // GAML_MF_GD_PROJECTION_H_
