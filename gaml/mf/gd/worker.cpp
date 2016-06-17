#include <algorithm>
#include <numeric>

#include "worker.h"

namespace gaml {

namespace mf {

namespace gd {

std::tuple<arma::fmat, arma::fmat> Worker::factor(const arma::sp_fmat pSlice,
                                                  const int pOffset,
                                                  const arma::sp_fmat uSlice,
                                                  const int uOffset,
                                                  const int k) {
  const int mbSize = this->minibatch;
  auto pTable = this->pTable;
  auto utTable = this->utTable;

  // Randomly initialize the factors
  this->randomizeTable(pTable, k, pSlice.n_rows, pOffset);
  this->randomizeTable(utTable, k, uSlice.n_cols, uOffset);

  petuum::PSTableGroup::GlobalBarrier();

  // Fetch the initial values
  arma::fmat P = this->loadMatrix(pTable, uSlice.n_rows, k);
  arma::fmat UT = this->loadMatrix(utTable, pSlice.n_cols, k);

  float step = 1.0;

  for (int i = 0; i < this->iterations; i++) {
    // Compute gradient for P
    arma::fmat pGrad(pSlice.n_rows, k, arma::fill::zeros);

    std::vector<int> mbIndices = this->selectMinibatch(pSlice, mbSize);
    arma::sp_fmat::const_iterator start = pSlice.begin();
    arma::sp_fmat::const_iterator end = pSlice.end();
    arma::sp_fmat::const_iterator it = start;
    for (int i = 0, j = 0; it != end && (mbSize == 0 || j < mbSize);
         ++it, ++i) {
      if (i == mbIndices[j]) {
        ++j;

        int row = it.row();
        int col = it.col();

        float diff = arma::dot(P.row(row + pOffset), UT.row(col)) - (*it);

        for (int x = 0; x < k; x++) {
          pGrad(row, x) += 2 * UT(col, x) * diff;
        }
      }
    }

    pGrad = arma::normalise(pGrad, 2, 1);
    pGrad = pGrad * (-step);

    // Update P table
    this->updateMatrixSlice(pGrad, pTable, pGrad.n_rows, pGrad.n_cols, pOffset);

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated P
    P = this->loadMatrix(pTable, uSlice.n_rows, k);

    // Project factor
    P = this->projection.project(P);

    // Compute gradient for U^T
    arma::fmat utGrad(uSlice.n_cols, k, arma::fill::zeros);

    mbIndices = this->selectMinibatch(uSlice, mbSize);
    start = uSlice.begin();
    end = uSlice.end();
    it = start;
    for (int i = 0, j = 0; it != end && (mbSize == 0 || j < mbSize);
         ++it, ++i) {
      if (i == mbIndices[j]) {
        ++j;

        int row = it.row();
        int col = it.col();

        float diff = arma::dot(P.row(row), UT.row(col + uOffset)) - (*it);

        for (int x = 0; x < k; x++) {
          utGrad(col, x) += 2 * P(row, x) * diff;
        }
      }
    }

    utGrad = arma::normalise(utGrad, 2, 1);
    utGrad = utGrad * (-step);

    // Update U table
    this->updateMatrixSlice(utGrad, utTable, utGrad.n_rows, utGrad.n_cols,
                            uOffset);

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated U^T
    UT = this->loadMatrix(utTable, pSlice.n_cols, k);

    // Project factor
    UT = this->projection.project(UT);

    step *= 0.9;
  }

  return {P, UT};
}

void Worker::initTables(int pTableId, int uTableId, int rowType, int k,
                        int pNumRows, int uNumRows) {
  petuum::ClientTableConfig p_config;
  p_config.table_info.row_type = rowType;
  p_config.table_info.row_capacity = pNumRows;
  p_config.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  p_config.table_info.table_staleness = 0;
  p_config.table_info.oplog_dense_serialized = true;
  p_config.table_info.dense_row_oplog_capacity =
      p_config.table_info.row_capacity;
  p_config.process_cache_capacity = k;
  p_config.oplog_capacity = k;
  p_config.thread_cache_capacity = 1;
  p_config.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(pTableId, p_config);

  petuum::ClientTableConfig u_config;
  u_config.table_info.row_type = rowType;
  u_config.table_info.row_capacity = uNumRows;
  u_config.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  u_config.table_info.table_staleness = 0;
  u_config.table_info.oplog_dense_serialized = true;
  u_config.table_info.dense_row_oplog_capacity =
      u_config.table_info.row_capacity;
  u_config.process_cache_capacity = k;
  u_config.oplog_capacity = k;
  u_config.thread_cache_capacity = 1;
  u_config.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(uTableId, u_config);
}

std::vector<int> Worker::selectMinibatch(const arma::sp_fmat& M,
                                         const int mbSize) {
  std::vector<int> indices(M.n_nonzero);
  std::iota(indices.begin(), indices.end(), 0);

  if (mbSize > 0) {
    std::shuffle(indices.begin(), indices.end(), this->rng);

    std::vector<int> minibatch(&indices[0], &indices[mbSize]);
    std::sort(minibatch.begin(), minibatch.end());

    return minibatch;
  } else {
    return indices;
  }
}
}
}
}
