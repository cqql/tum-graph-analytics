#include "worker.h"
#include "../util/table.h"

namespace gaml {

namespace biases {

std::tuple<float, arma::fvec, arma::fvec, arma::sp_fmat, arma::sp_fmat>
Worker::compute(const arma::sp_fmat pSlice, const int pOffset,
                const arma::sp_fmat uSlice, const int uOffset) {
  const int nranks = this->nranks;
  const int rank = this->rank;
  auto pTable = this->pTable;
  auto uTable = this->uTable;
  auto meanTable = this->meanTable;

  // Create local mutable copies for subtracting means etc.
  arma::sp_fmat pslice(pSlice);
  arma::sp_fmat uslice(uSlice);

  // Store the number of non-zero entries in this P slice in the first row
  meanTable.Inc(0, rank, pslice.n_nonzero);

  // Store their sum in the second row
  const arma::fvec nzvals(pslice.values, pslice.n_nonzero);
  meanTable.Inc(1, rank, arma::sum(nzvals));

  petuum::PSTableGroup::GlobalBarrier();

  // Load the values to compute the global mean
  const arma::fmat meansMat =
      gaml::util::table::loadMatrix(meanTable, nranks, 2);
  const arma::fvec nnzs = meansMat.col(0);
  const arma::fvec sums = meansMat.col(1);
  const float mean = arma::sum(sums) / arma::sum(nnzs);

  // Subtract the global mean from all non-zero entries
  pslice = this->subtract(pslice, mean);
  uslice = this->subtract(uslice, mean);

  // Compute the user specific means for this U slice
  arma::fvec usliceMeans(uslice.n_cols, arma::fill::zeros);
  for (int i = 0; i < uslice.n_cols; i++) {
    const int nnz = uslice.col_ptrs[i + 1] - uslice.col_ptrs[i];

    if (nnz > 0) {
      usliceMeans(i) =
          arma::mean(arma::fvec(&uslice.values[uslice.col_ptrs[i]], nnz));
    }
  }

  // Share the user-specific biases
  gaml::util::table::updateMatrixSlice(usliceMeans, uTable, uslice.n_cols, 1,
                                       uOffset);

  // Center the columns of U slice
  uslice = this->subtractColumnwise(uslice, usliceMeans);

  petuum::PSTableGroup::GlobalBarrier();

  // Load the complete user biases
  const arma::fvec uMeans =
      gaml::util::table::loadMatrix(uTable, pslice.n_cols, 1).col(0);

  // Center the columns of P slice
  pslice = this->subtractColumnwise(pslice, uMeans);

  // Compute the product-specific biases for this P slice
  const arma::sp_fmat psliceT = pslice.t();
  arma::fvec psliceMeans(psliceT.n_cols, arma::fill::zeros);
  for (int i = 0; i < psliceT.n_cols; i++) {
    const int nnz = psliceT.col_ptrs[i + 1] - psliceT.col_ptrs[i];

    if (nnz > 0) {
      psliceMeans(i) =
          arma::mean(arma::fvec(&psliceT.values[psliceT.col_ptrs[i]], nnz));
    }
  }

  // Share the product-specific biases
  gaml::util::table::updateMatrixSlice(psliceMeans, pTable, pslice.n_rows, 1,
                                       pOffset);

  // Center the rows of P slice
  pslice = this->subtractRowwise(pslice, psliceMeans);

  petuum::PSTableGroup::GlobalBarrier();

  // Load the complete user biases
  const arma::fvec pMeans =
      gaml::util::table::loadMatrix(pTable, uslice.n_rows, 1).col(0);

  // Center the rows of U slice
  uslice = this->subtractRowwise(uslice, pMeans);

  return {mean, uMeans, pMeans, uslice, pslice};
}

void Worker::initTables(int rowType, int pTableId, int pNumRows, int uTableId,
                        int uNumRows, int meanTableId, int nranks) {
  petuum::ClientTableConfig pConfig;
  pConfig.table_info.row_type = rowType;
  pConfig.table_info.row_capacity = pNumRows;
  pConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  pConfig.table_info.table_staleness = 0;
  pConfig.table_info.oplog_dense_serialized = true;
  pConfig.table_info.dense_row_oplog_capacity = pConfig.table_info.row_capacity;
  pConfig.process_cache_capacity = 1;
  pConfig.oplog_capacity = 1;
  pConfig.thread_cache_capacity = 1;
  pConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(pTableId, pConfig);

  petuum::ClientTableConfig uConfig;
  uConfig.table_info.row_type = rowType;
  uConfig.table_info.row_capacity = uNumRows;
  uConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  uConfig.table_info.table_staleness = 0;
  uConfig.table_info.oplog_dense_serialized = true;
  uConfig.table_info.dense_row_oplog_capacity = uConfig.table_info.row_capacity;
  uConfig.process_cache_capacity = 1;
  uConfig.oplog_capacity = 1;
  uConfig.thread_cache_capacity = 1;
  uConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(uTableId, uConfig);

  petuum::ClientTableConfig meanConfig;
  meanConfig.table_info.row_type = rowType;
  meanConfig.table_info.row_capacity = nranks;
  meanConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  meanConfig.table_info.table_staleness = 0;
  meanConfig.table_info.oplog_dense_serialized = true;
  meanConfig.table_info.dense_row_oplog_capacity =
      meanConfig.table_info.row_capacity;
  meanConfig.process_cache_capacity = 2;
  meanConfig.oplog_capacity = 2;
  meanConfig.thread_cache_capacity = 1;
  meanConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(meanTableId, meanConfig);
}

arma::sp_fmat Worker::subtract(const arma::sp_fmat& A, const float& b) {
  const arma::uvec rowinds(A.row_indices, A.n_nonzero);
  const arma::uvec colptrs(A.col_ptrs, A.n_cols + 1);
  arma::fvec vals(A.values, A.n_nonzero);

  vals -= b * arma::ones<arma::fvec>(A.n_nonzero);

  return arma::sp_fmat(rowinds, colptrs, vals, A.n_rows, A.n_cols);
}

arma::sp_fmat Worker::subtractColumnwise(const arma::sp_fmat& A,
                                         const arma::fvec& b) {
  const arma::uvec rowinds(A.row_indices, A.n_nonzero);
  const arma::uvec colptrs(A.col_ptrs, A.n_cols + 1);
  arma::fvec vals(A.values, A.n_nonzero);

  arma::uword col = 0;
  arma::uword next = 1;
  for (int i = 0; i < A.n_nonzero; i++) {
    if (i >= colptrs[next]) {
      col++;
      next++;
    }

    vals[i] -= b[col];
  }

  return arma::sp_fmat(rowinds, colptrs, vals, A.n_rows, A.n_cols);
}

arma::sp_fmat Worker::subtractRowwise(const arma::sp_fmat& A,
                                      const arma::fvec& b) {
  const arma::uvec rowinds(A.row_indices, A.n_nonzero);
  const arma::uvec colptrs(A.col_ptrs, A.n_cols + 1);
  arma::fvec vals(A.values, A.n_nonzero);

  for (int i = 0; i < A.n_nonzero; i++) {
    vals[i] -= b[rowinds[i]];
  }

  return arma::sp_fmat(rowinds, colptrs, vals, A.n_rows, A.n_cols);
}
}
}
