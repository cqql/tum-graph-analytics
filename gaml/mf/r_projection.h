#ifndef R_PROJECTION_H_
#define R_PROJECTION_H_

#include "projection.h"

namespace gaml {

namespace mf {

class RProjection : public Projection {
  arma::fmat project(arma::fmat A) const;
};
}
}

#endif  // R_PROJECTION_H_
