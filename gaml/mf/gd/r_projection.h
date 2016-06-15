#ifndef GAML_MF_GD_R_PROJECTION_H_
#define GAML_MF_GD_R_PROJECTION_H_

#include "projection.h"

namespace gaml {

namespace mf {

namespace gd {

class RProjection : public Projection {
  arma::fmat project(arma::fmat A) const;
};
}
}
}

#endif  // GAML_MF_GD_R_PROJECTION_H_
