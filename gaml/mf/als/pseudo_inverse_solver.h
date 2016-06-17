#ifndef GAML_MF_ALS_PSEUDO_INVERSE_SOLVER_H_
#define GAML_MF_ALS_PSEUDO_INVERSE_SOLVER_H_

#include <armadillo>

#include "solver.h"

namespace gaml {

namespace mf {

namespace als {

class PseudoInverseSolver : public Solver {
 public:
  arma::fmat solve(const arma::fmat& P, const arma::sp_fmat& R);
};
}
}
}

#endif  // GAML_MF_ALS_PSEUDO_INVERSE_SOLVER_H_
