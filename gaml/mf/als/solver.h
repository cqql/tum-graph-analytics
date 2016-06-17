#ifndef GAML_MF_ALS_SOLVER_H_
#define GAML_MF_ALS_SOLVER_H_

namespace gaml {

namespace mf {

namespace als {

/**
 * Solves P*U = R for U for some sparse R
 */
class Solver {
 public:
  virtual arma::fmat solve(const arma::fmat& P, const arma::sp_fmat& R) = 0;
};
}
}
}

#endif  // GAML_MF_ALS_SOLVER_H_
