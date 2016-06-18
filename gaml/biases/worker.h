#ifndef GAML_BIASES_WORKER_H_
#define GAML_BIASES_WORKER_H_

#include <tuple>

#include <armadillo>
#include <petuum_ps_common/include/petuum_ps.hpp>

namespace gaml {

namespace biases {

class Worker {
 public:
  Worker(const int pTableId, const int uTableId, const int meanTableId,
         const int nranks, const int rank)
      : pTable(petuum::PSTableGroup::GetTableOrDie<float>(pTableId)),
        uTable(petuum::PSTableGroup::GetTableOrDie<float>(uTableId)),
        meanTable(petuum::PSTableGroup::GetTableOrDie<float>(meanTableId)),
        nranks(nranks),
        rank(rank){};

  std::tuple<float, arma::fvec, arma::fvec, arma::sp_fmat, arma::sp_fmat>
  compute(const arma::sp_fmat pSlice, const int pOffset,
          const arma::sp_fmat uSlice, const int uOffset);

  static void initTables(int rowType, int pTableId, int pNumRows, int uTableId,
                         int uNumRows, int meanTableId, int nranks);

 private:
  petuum::Table<float> pTable;
  petuum::Table<float> uTable;
  petuum::Table<float> meanTable;
  const int nranks;
  const int rank;

  /**
   * Subtract b from the non-zero entries of A in-place
   */
  arma::sp_fmat subtract(const arma::sp_fmat& A, const float& b);

  /**
   * Subtract the i-th value of b from the non-zero entries in the i-th column
   * of A in-place
   */
  arma::sp_fmat subtractColumnwise(const arma::sp_fmat& A, const arma::fvec& b);

  /**
   * Subtract the i-th value of b from the non-zero entries in the i-th row
   * of A in-place
   */
  arma::sp_fmat subtractRowwise(const arma::sp_fmat& A, const arma::fvec& b);
};
}
}

#endif  // GAML_BIASES_WORKER_H_
