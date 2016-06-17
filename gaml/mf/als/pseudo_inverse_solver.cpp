#include "pseudo_inverse_solver.h"

namespace gaml {

namespace mf {

namespace als {

arma::fmat PseudoInverseSolver::solve(const arma::fmat& P,
                                      const arma::sp_fmat& R) {
  arma::fmat U(P.n_cols, R.n_cols);

  for (int i = 0; i < R.n_cols; i++) {
    // Number of non-zero values in column i
    const arma::uword nnz = R.col_ptrs[i + 1] - R.col_ptrs[i];
    // Row indices of non-zero values in column i
    const auto nonzeros = arma::uvec(&R.row_indices[R.col_ptrs[i]], nnz);
    // Non-zero entries of column i
    const auto nnzcol = arma::fvec(&R.values[R.col_ptrs[i]], nnz);

    U.col(i) = arma::pinv(P.rows(nonzeros)) * nnzcol;
  }

  return U;
}
}
}
}
