#ifndef NN_PROJECTION_H_
#define NN_PROJECTION_H_

#include "projection.h"

namespace gaml {

namespace mf {

/**
 * Non-negative projection
 */
class NNProjection : public Projection {
  arma::fmat project(arma::fmat A) const;
};
}
}

#endif  // NN_PROJECTION_H_
