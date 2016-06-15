#include "nn_projection.h"

namespace gaml {

namespace mf {

namespace gd {

arma::fmat NNProjection::project(arma::fmat U) const {
  return arma::clamp(U, 0.0, U.max());
}
}
}
}
