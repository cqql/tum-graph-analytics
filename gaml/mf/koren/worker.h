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
         const float lambdaqpy, const float gammab, const float gammaqpy,
         const float atol, const float rtol, const int muTableId,
         const int biTableId, const int buTableId, const int qTableId,
         const int pTableId, const int yTableId, const int errorTableId,
         const int nruTableId, const int ruTableId, const int seTableId)
      : rank(rank),
        nranks(nranks),
        lambdab(lambdab),
        lambdaqpy(lambdaqpy),
        gammab(gammab),
        gammaqpy(gammaqpy),
        atol(atol),
        rtol(rtol),
        muTable(petuum::PSTableGroup::GetTableOrDie<float>(muTableId)),
        biTable(petuum::PSTableGroup::GetTableOrDie<float>(biTableId)),
        buTable(petuum::PSTableGroup::GetTableOrDie<float>(buTableId)),
        qTable(petuum::PSTableGroup::GetTableOrDie<float>(qTableId)),
        pTable(petuum::PSTableGroup::GetTableOrDie<float>(pTableId)),
        yTable(petuum::PSTableGroup::GetTableOrDie<float>(yTableId)),
        errorTable(petuum::PSTableGroup::GetTableOrDie<float>(errorTableId)),
        nruTable(petuum::PSTableGroup::GetTableOrDie<int>(nruTableId)),
        ruTable(petuum::PSTableGroup::GetTableOrDie<int>(ruTableId)),
        seTable(petuum::PSTableGroup::GetTableOrDie<float>(seTableId)) {}

  std::tuple<float, arma::fvec, arma::fvec, arma::fmat, arma::fmat, arma::fmat>
  factor(const arma::sp_fmat iSlice, const int iOffset,
         const arma::sp_fmat uSlice, const int uOffset, const int k);

  static void initTables(int muTableId, int biTableId, int buTableId,
                         int qTableId, int pTableId, int yTableId,
                         int errorTableId, int nruTableId, int ruTableId,
                         int seTableId, int floatRowType, int intRowType, int k,
                         int nnz, int maxFill, int nItems, int nUsers,
                         int nranks);

 private:
  const int rank;
  const int nranks;
  const float lambdab;
  const float lambdaqpy;
  const float gammab;
  const float gammaqpy;
  const float atol;
  const float rtol;
  petuum::Table<float> muTable;
  petuum::Table<float> biTable;
  petuum::Table<float> buTable;
  petuum::Table<float> qTable;
  petuum::Table<float> pTable;
  petuum::Table<float> yTable;
  petuum::Table<float> errorTable;
  petuum::Table<int> nruTable;
  petuum::Table<int> ruTable;
  petuum::Table<float> seTable;

  void stepTable(petuum::Table<float>& table, const arma::fmat& step,
                 int rowOffset, int colOffset);
};
}
}
}

#endif  // GAML_MF_KOREN_WORKER_H_
