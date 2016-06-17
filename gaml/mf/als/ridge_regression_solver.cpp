#include "ridge_regression_solver.h"

namespace gaml {

namespace mf {

namespace als {

arma::fmat RidgeRegressionSolver::solve(const arma::fmat& P,
                                        const arma::sp_fmat& R) {
  arma::fmat U(P.n_cols, R.n_cols);
  const arma::fmat Gamma =
      (this->alpha * arma::fmat(P.n_cols, P.n_cols, arma::fill::eye));

  for (int i = 0; i < R.n_cols; i++) {
    // Number of non-zero values in column i
    const arma::uword nnz = R.col_ptrs[i + 1] - R.col_ptrs[i];
    // Row indices of non-zero values in column i
    const arma::uvec nonzeros(&R.row_indices[R.col_ptrs[i]], nnz);
    // Non-zero entries of column i
    const arma::fvec nnzcol(&R.values[R.col_ptrs[i]], nnz);

    const arma::fmat A = P.rows(nonzeros);
    const arma::fvec b = nnzcol;

    const arma::fmat A2 = A.t() * A + Gamma;
    const arma::fvec b2 = A.t() * b;

    U.col(i) = arma::solve(A2, b2);
  }

  return U;
}
}
}
}
