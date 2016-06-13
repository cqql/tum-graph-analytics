#ifndef PROJECTION_H_
#define PROJECTION_H_

#include <armadillo>

namespace gaml {

namespace mf {

class Projection {
 public:
  virtual arma::fmat project(arma::fmat) const = 0;
};
}
}

#endif  // PROJECTION_H_
