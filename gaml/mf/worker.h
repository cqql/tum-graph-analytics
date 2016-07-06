#ifndef GAML_MF_WORKER_H_
#define GAML_MF_WORKER_H_

#include <tuple>

#include <armadillo>
#include <petuum_ps_common/include/petuum_ps.hpp>

namespace gaml {

namespace mf {

class Worker {
 public:
  Worker(const int pTableId, const int uTableId)
      : pTableId(pTableId),
        uTableId(uTableId),
        pTable(petuum::PSTableGroup::GetTableOrDie<float>(pTableId)),
        utTable(petuum::PSTableGroup::GetTableOrDie<float>(uTableId)) {}

  virtual std::tuple<arma::fmat, arma::fmat> factor(const arma::sp_fmat pSlice,
                                                    const int pOffset,
                                                    const arma::sp_fmat uSlice,
                                                    const int uOffset,
                                                    const int k) = 0;

 protected:
  const int pTableId;
  const int uTableId;
  petuum::Table<float> pTable;
  petuum::Table<float> utTable;
};
}
}

#endif  // GAML_MF_WORKER_H_
