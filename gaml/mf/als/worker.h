#ifndef GAML_MF_ALS_WORKER_H_
#define GAML_MF_ALS_WORKER_H_

#include "../worker.h"
#include "logger.h"
#include "pseudo_inverse_solver.h"

namespace gaml {

namespace mf {

namespace als {

class Worker : public gaml::mf::Worker {
 public:
  Worker(const int pTableId, const int uTableId, const int seTableId,
         const int nranks, const int rank, const float atol, const float rtol)
      : gaml::mf::Worker(pTableId, uTableId),
        seTable(petuum::PSTableGroup::GetTableOrDie<float>(seTableId)),
        nranks(nranks),
        rank(rank),
        atol(atol),
        rtol(rtol),
        solver(new PseudoInverseSolver()){}

  std::tuple<arma::fmat, arma::fmat> factor(const arma::sp_fmat pSlice,
                                            const int pOffset,
                                            const arma::sp_fmat uSlice,
                                            const int uOffset, const int k);

  static void initTables(int pTableId, int uTableId, int rowType, int k,
                         int pNumRows, int uNumRows, int seTableId,
                         int nranks);

 private:
  petuum::Table<float> seTable;
  const int nranks;
  const int rank;
  const float atol;
  const float rtol;
  Logger logger;
  std::unique_ptr<Solver> solver;
};
}
}
}

#endif  // GAML_MF_ALS_WORKER_H_
