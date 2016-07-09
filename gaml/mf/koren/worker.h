#ifndef GAML_MF_KOREN_WORKER_H_
#define GAML_MF_KOREN_WORKER_H_

#include <tuple>

#include <armadillo>
#include <petuum_ps_common/include/petuum_ps.hpp>

namespace gaml {

namespace mf {

namespace koren {

class Worker {
 public:
  Worker(const int rank, const int nranks, const float lambdab,
         const float lambdaqpy, const float gammab, const float gammat,
         const float gammaqpy, const float beta, const float atol,
         const float rtol, const int tiTableId, const int muTableId,
         const int biTableId, const int buTableId, const int alphaTableId,
         const int kappaTableId, const int qTableId, const int pTableId,
         const int yTableId, const int seTableId)
      : rank(rank),
        nranks(nranks),
        lambdab(lambdab),
        lambdaqpy(lambdaqpy),
        gammab(gammab),
        gammat(gammat),
        gammaqpy(gammaqpy),
        beta(beta),
        atol(atol),
        rtol(rtol),
        tiTable(petuum::PSTableGroup::GetTableOrDie<float>(tiTableId)),
        muTable(petuum::PSTableGroup::GetTableOrDie<float>(muTableId)),
        biTable(petuum::PSTableGroup::GetTableOrDie<float>(biTableId)),
        buTable(petuum::PSTableGroup::GetTableOrDie<float>(buTableId)),
        alphaTable(petuum::PSTableGroup::GetTableOrDie<float>(alphaTableId)),
        kappaTable(petuum::PSTableGroup::GetTableOrDie<float>(kappaTableId)),
        qTable(petuum::PSTableGroup::GetTableOrDie<float>(qTableId)),
        pTable(petuum::PSTableGroup::GetTableOrDie<float>(pTableId)),
        yTable(petuum::PSTableGroup::GetTableOrDie<float>(yTableId)),
        seTable(petuum::PSTableGroup::GetTableOrDie<float>(seTableId)) {}

  std::tuple<float, arma::fvec, arma::fvec, arma::fvec, arma::fvec, arma::fmat,
             arma::fmat, arma::fmat>
  factor(const arma::sp_fmat iSlice, const int iOffset,
         const arma::sp_fmat uSlice, const arma::sp_fmat tSlice,
         const int uOffset, const int k);

  static void initTables(int tiTableId, int muTableId, int biTableId,
                         int buTableId, int alphaTableId, int kappaTableId,
                         int qTableId, int pTableId, int yTableId,
                         int seTableId, int floatRowType, int intRowType, int k,
                         int nItems, int nUsers, int nranks);

 private:
  const int rank;
  const int nranks;
  const float lambdab;
  const float lambdaqpy;
  const float gammab;
  const float gammat;
  const float gammaqpy;
  const float beta;
  const float atol;
  const float rtol;
  petuum::Table<float> tiTable;
  petuum::Table<float> muTable;
  petuum::Table<float> biTable;
  petuum::Table<float> buTable;
  petuum::Table<float> alphaTable;
  petuum::Table<float> kappaTable;
  petuum::Table<float> qTable;
  petuum::Table<float> pTable;
  petuum::Table<float> yTable;
  petuum::Table<float> seTable;

  void stepTable(petuum::Table<float>& table, const arma::fmat& step,
                 int rowOffset, int colOffset);
};
}
}
}

#endif  // GAML_MF_KOREN_WORKER_H_
