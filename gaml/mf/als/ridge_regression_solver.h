#ifndef GAML_MF_ALS_RIDGE_REGRESSION_SOLVER_H_
#define GAML_MF_ALS_RIDGE_REGRESSION_SOLVER_H_

#include <armadillo>

#include "solver.h"

namespace gaml {

namespace mf {

namespace als {

class RidgeRegressionSolver : public Solver {
 public:
  RidgeRegressionSolver(const float alpha) : alpha(alpha) {}

  arma::fmat solve(const arma::fmat& P, const arma::sp_fmat& R);

 private:
  const float alpha;
};
}
}
}

#endif  // GAML_MF_ALS_RIDGE_REGRESSION_SOLVER_H_
