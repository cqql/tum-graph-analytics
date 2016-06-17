#include "worker.h"

namespace gaml {

namespace mf {

namespace als {

std::tuple<arma::fmat, arma::fmat> Worker::factor(const arma::sp_fmat pSlice,
                                                  const int pOffset,
                                                  const arma::sp_fmat uSlice,
                                                  const int uOffset,
                                                  const int k) {
  const float atol = this->atol;
  const float rtol = this->rtol;
  auto pTable = this->pTable;
  auto utTable = this->utTable;

  // Randomly initialize the factors
  this->randomizeTable(pTable, k, pSlice.n_rows, pOffset);
  this->randomizeTable(utTable, k, uSlice.n_cols, uOffset);

  // Share number of non-zero entries in P slice
  this->seTable.Inc(1, this->rank, pSlice.n_nonzero);

  petuum::PSTableGroup::GlobalBarrier();

  // Fetch the initial values
  auto P = this->loadMatrix(pTable, uSlice.n_rows, k);
  auto UT = this->loadMatrix(utTable, pSlice.n_cols, k);

  // We want to access the rows of pSlice directly in the loop which is
  // preferably done by accessing the columns of the transposed version due to
  // the CCS storage format
  const auto pSliceT = pSlice.t().eval();

  float prevSE = 0.0;
  float prevMSE = 0.0;
  float aimprov = 0.0;
  float rimprov = 0.0;

  int iteration = 0;
  do {
    iteration++;

    /////////////////////
    // Update P

    // Eval this because armadillo throws an error otherwise
    auto Pnew = this->solver->solve(UT, pSliceT).t().eval();
    // Eval this for colptr()
    auto Pupdate =
        (Pnew - P.submat(pOffset, 0, pOffset + pSlice.n_rows - 1, k - 1))
            .eval();

    // Update P table
    this->updateMatrixSlice(Pupdate, pTable, pSlice.n_rows, k, pOffset);

    petuum::PSTableGroup::GlobalBarrier();

    // Load updated P
    P = this->loadMatrix(pTable, uSlice.n_rows, k);

    ////////////////////
    // Update U^T

    // Eval this because armadillo throws an error otherwise
    auto UTnew = this->solver->solve(P, uSlice).t().eval();
    // Eval this for colptr()
    auto UTupdate =
        (UTnew - UT.submat(uOffset, 0, uOffset + uSlice.n_cols - 1, k - 1))
            .eval();

    // Update U table
    this->updateMatrixSlice(UTupdate, utTable, uSlice.n_cols, k, uOffset);

    petuum::PSTableGroup::GlobalBarrier();

    // Load updated U^T
    UT = this->loadMatrix(utTable, pSlice.n_cols, k);

    ///////////////////////
    // Compute SE (squared error) for this P slice

    float se = 0.0;

    arma::sp_fmat::const_iterator start = pSlice.begin();
    arma::sp_fmat::const_iterator end = pSlice.end();
    for (arma::sp_fmat::const_iterator it = start; it != end; ++it) {
      int row = pOffset + it.row();
      int col = it.col();
      float diff = arma::dot(P.row(row), UT.row(col)) - (*it);
      se += diff * diff;
    }

    // Update SE
    this->seTable.Inc(0, this->rank, se - prevSE);
    prevSE = se;
    petuum::PSTableGroup::GlobalBarrier();

    // Fetch all updated SEs
    const auto SEs = this->loadMatrix(this->seTable, this->nranks, 2);

    // Compute MSE
    const float mse = arma::sum(SEs.col(0)) / arma::sum(SEs.col(1));

    // Absolute and relative improvements
    aimprov = prevMSE - mse;
    rimprov = aimprov / prevMSE;
    prevMSE = mse;

    if (this->rank == 0) {
      this->logger.log(iteration, mse, aimprov, rimprov);
    }
  } while (iteration == 1 || rimprov >= rtol && aimprov >= atol);

  return {P, UT};
}

void Worker::initTables(int pTableId, int uTableId, int rowType, int k,
                        int pNumRows, int uNumRows, int seTableId, int nranks) {
  petuum::ClientTableConfig pConfig;
  pConfig.table_info.row_type = rowType;
  pConfig.table_info.row_capacity = pNumRows;
  pConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  pConfig.table_info.table_staleness = 0;
  pConfig.table_info.oplog_dense_serialized = true;
  pConfig.table_info.dense_row_oplog_capacity = pConfig.table_info.row_capacity;
  pConfig.process_cache_capacity = k;
  pConfig.oplog_capacity = k;
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
  uConfig.process_cache_capacity = k;
  uConfig.oplog_capacity = k;
  uConfig.thread_cache_capacity = 1;
  uConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(uTableId, uConfig);

  petuum::ClientTableConfig seConfig;
  seConfig.table_info.row_type = rowType;
  seConfig.table_info.row_capacity = nranks;
  seConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  seConfig.table_info.table_staleness = 0;
  seConfig.table_info.oplog_dense_serialized = true;
  seConfig.table_info.dense_row_oplog_capacity =
      seConfig.table_info.row_capacity;
  seConfig.process_cache_capacity = 2;
  seConfig.oplog_capacity = 2;
  seConfig.thread_cache_capacity = 1;
  seConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(seTableId, seConfig);
}
}
}
}
