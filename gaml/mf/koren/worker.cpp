#include <vector>

#include "../../util/table.h"
#include "worker.h"

namespace gaml {

namespace mf {

namespace koren {

// Subtract b from all non-zero entries of A
arma::sp_fmat operator-(const arma::sp_fmat& A, const float& b) {
  const arma::uvec rowinds(A.row_indices, A.n_nonzero);
  const arma::uvec colptrs(A.col_ptrs, A.n_cols + 1);
  arma::fvec vals(A.values, A.n_nonzero);

  vals -= b * arma::ones<arma::fvec>(A.n_nonzero);

  return arma::sp_fmat(rowinds, colptrs, vals, A.n_rows, A.n_cols);
}

std::tuple<float, arma::fvec, arma::fvec, arma::fmat, arma::fmat, arma::fmat>
Worker::factor(const arma::sp_fmat iSlice, const int iOffset,
               const arma::sp_fmat uSlice, const int uOffset, const int k) {
  const auto rank = this->rank;
  const auto nranks = this->nranks;
  const auto lambdab = this->lambdab;
  const auto lambdaqpy = this->lambdaqpy;
  auto gammab = this->gammab;
  auto gammaqpy = this->gammaqpy;
  const auto atol = this->atol;
  const auto rtol = this->rtol;
  auto muTable = this->muTable;
  auto biTable = this->biTable;
  auto buTable = this->buTable;
  auto qTable = this->qTable;
  auto yTable = this->yTable;
  auto pTable = this->pTable;
  auto errorTable = this->errorTable;
  auto nruTable = this->nruTable;
  auto ruTable = this->ruTable;
  auto seTable = this->seTable;

  const auto nItems = uSlice.n_rows;
  const auto nUsers = iSlice.n_cols;

  // Number of things in the local slices
  const auto nItemsLocal = iSlice.n_rows;
  const auto nUsersLocal = uSlice.n_cols;

  // Compute mean of the U slice

  // Store the number of non-zero entries in this u slice in the first row
  muTable.Inc(0, rank, uSlice.n_nonzero);

  // Store their sum in the second row
  const arma::fvec uNzVals(uSlice.values, uSlice.n_nonzero);
  muTable.Inc(1, rank, arma::sum(uNzVals));

  petuum::PSTableGroup::GlobalBarrier();

  // Compute the global mean
  const arma::fmat muMat = gaml::util::table::loadMatrix(muTable, nranks, 2);
  const float totalSum = arma::sum(muMat.col(1));
  const float nnz = arma::sum(muMat.col(0));
  const float mu = totalSum / nnz;

  // Normalize the local slices
  const arma::sp_fmat nISlice = iSlice - mu;
  const arma::sp_fmat nUSlice = uSlice - mu;

  // Initialize all the parameters
  arma::fvec bi(nItems, arma::fill::zeros);
  arma::fvec bu(nUsers, arma::fill::zeros);
  arma::fmat q(k, nItems, arma::fill::randn);
  arma::fmat p(k, nUsers, arma::fill::randn);
  arma::fmat y(k, nItems, arma::fill::zeros);

  // TODO: Maybe store randomized P and U in tables??

  // Indices of all items rated by u for each user u in the local U slice
  std::vector<arma::uvec> RuLocal(nUsersLocal);
  // Number of items rated by each u
  arma::uvec nRuLocal(nUsersLocal);

  for (int i = 0; i < nUsersLocal; ++i) {
    // Number of non-zero entries in this column
    const int nnz = nUSlice.col_ptrs[i + 1] - nUSlice.col_ptrs[i];

    RuLocal[i] = arma::uvec(&nUSlice.row_indices[nUSlice.col_ptrs[i]], nnz);
    nRuLocal(i) = nnz;
  }

  // And their inverse square roots (i.e. ^-1/2)
  const arma::uvec isqNRuLocal = 1 / arma::sqrt(nRuLocal);

  // Finally the same for all items in the local I slice
  std::vector<arma::uvec> RiLocal(nItemsLocal);
  // Number of users that rated item i
  arma::uvec nRiLocal(nItemsLocal);

  // Transpose the I slice so that we can easily compute this
  const arma::sp_fmat nISliceT = nISlice.t();
  for (int i = 0; i < nItemsLocal; ++i) {
    // Number of non-zero entries in this column
    const int nnz = nISliceT.col_ptrs[i + 1] - nISliceT.col_ptrs[i];

    RiLocal[i] = arma::uvec(&nISliceT.row_indices[nISliceT.col_ptrs[i]], nnz);
    nRiLocal(i) = nnz;
  }

  // Synchronize Ru and nRu
  for (int u = 0; u < nUsersLocal; u++) {
    nruTable.Inc(0, uOffset + u, nRuLocal(u));

    petuum::DenseUpdateBatch<int> batch(0, nRuLocal(u));
    for (int j = 0; j < nRuLocal(u); ++j) {
      // Copy manually to convert from arma::uword to int
      batch[j] = (int)RuLocal[u](j);
    }
    ruTable.DenseBatchInc(u + uOffset, batch);
  }

  petuum::PSTableGroup::GlobalBarrier();

  // Load global nRu and Ru
  arma::uvec nRu(nUsers);
  {
    std::vector<int> buf;
    petuum::RowAccessor rowacc;
    const auto& col = nruTable.Get<petuum::DenseRow<int>>(0, &rowacc);
    col.CopyToVector(&buf);

    // Copy manually again to convert back to arma::uword
    for (int j = 0; j < nUsers; ++j) {
      nRu(j) = (arma::uword)buf[j];
    }
  }

  std::vector<arma::uvec> Ru(nUsers);

  for (int u = 0; u < nUsers; ++u) {
    std::vector<int> buf;
    petuum::RowAccessor rowacc;
    const auto& col = ruTable.Get<petuum::DenseRow<int>>(u, &rowacc);
    col.CopyToVector(&buf);

    Ru[u] = arma::uvec(nRu(u));

    // Copy manually again to convert back to arma::uword
    for (int j = 0; j < nRu(u); ++j) {
      Ru[u](j) = (arma::uword)buf[j];
    }
  }

  const arma::uvec isqNRu = 1 / arma::sqrt(nRu);

  // nRu and Ru actually tell us the coordinates of all entries of R. We will
  // use this later to synchronize the global error values.

  // Column pointers into E in CSC layout
  arma::uvec eColPtrs = arma::shift(arma::cumsum(nRu), 1);
  eColPtrs(0) = 0;

  // Total number of entries in earlier slices
  const arma::uword uEntryOffset = eColPtrs(uOffset);

  // Cache the locations of entries in the local U slice
  arma::umat uLocations(2, nUSlice.n_nonzero);
  {
    int index = 0;
    for (arma::sp_fmat::const_iterator it = nUSlice.begin(),
                                       end = nUSlice.end();
         it != end; ++it, ++index) {
      uLocations(0, index) = it.row();
      uLocations(1, index) = it.col();
    }
  }

  // Cache the locations of entries in the local I slice
  arma::umat iLocations(2, nISlice.n_nonzero);
  {
    int index = 0;
    for (arma::sp_fmat::const_iterator it = nISlice.begin(),
                                       end = nISlice.end();
         it != end; ++it, ++index) {
      iLocations(0, index) = it.row();
      iLocations(1, index) = it.col();
    }
  }

  // Values from previous iteration to compute the diff against for updates
  arma::fvec uErrorsPrev(nUSlice.n_nonzero, arma::fill::zeros);
  float prevSE = 0.0;
  float prevMSE = 0.0;
  float aimprov = 0.0;
  float rimprov = 0.0;

  // Run the gradient descent
  int iteration = 0;
  do {
    iteration++;

    // Prediction error matrix for the U slice
    arma::fvec uErrors(nUSlice.n_nonzero);
    {
      int index = 0;
      for (arma::sp_fmat::const_iterator it = nUSlice.begin(),
                                         end = nUSlice.end();
           it != end; ++it, ++index) {
        const arma::uword i = it.row();
        const arma::uword u = it.col();

        uErrors(index) =
            *it - bi(i) - bu(u) -
            arma::dot(q.col(i),
                      p.col(u + uOffset) +
                          isqNRuLocal(u) * arma::sum(y.cols(RuLocal[u]), 1));
      }
    }
    const arma::sp_fmat uE(uLocations, uErrors);

    // Synchronize the global E matrix with our local uE slice
    {
      const arma::fmat uErrorsDiff = uErrors - uErrorsPrev;
      petuum::DenseUpdateBatch<float> batch(uEntryOffset, nUSlice.n_nonzero);
      std::memcpy(batch.get_mem(), uErrorsDiff.memptr(),
                  nUSlice.n_nonzero * sizeof(float));
      errorTable.DenseBatchInc(0, batch);
      uErrorsPrev = uErrors;
    }

    petuum::PSTableGroup::GlobalBarrier();

    // Load the submatrix of E that coincides with our local I slice
    arma::fvec iErrors(nISlice.n_nonzero);
    {
      std::vector<float> buf;
      petuum::RowAccessor rowacc;
      const auto& col = errorTable.Get<petuum::DenseRow<float>>(0, &rowacc);
      col.CopyToVector(&buf);

      for (int j = 0; j < nISlice.n_nonzero; ++j) {
        const int i = iLocations(0, j) + iOffset;
        const int u = iLocations(1, j);

        // buf is E in CSC storage format, so we look into the u-th column and
        // then find entry in the compressed column that corresponds to row i
        for (int a = 0; a < nRu(u); ++a) {
          if (Ru[u](a) == i) {
            iErrors(j) = buf[eColPtrs[u] + a];
            goto CONTINUE_OUTER;
          }
        }

        std::cout << "Did not find entry (" << i << ", " << u << ") in E"
                  << std::endl;

      CONTINUE_OUTER:;
      }
    }
    const arma::sp_fmat iE(iLocations, iErrors);

    // Compute gradients of the locally managed parameters
    const arma::fvec biGrad =
        2 * (lambdab * nRiLocal % bi.rows(iOffset, iOffset + nItemsLocal - 1) -
             arma::sum(iE, 1));
    const arma::fvec buGrad =
        2 * (lambdab * nRuLocal % bu.rows(uOffset, uOffset + nUsersLocal - 1) -
             arma::sum(uE, 0).t());

    arma::fmat qGrad(k, nItemsLocal);
    arma::fmat pGrad(k, nUsersLocal);
    arma::fmat yGrad(k, nItemsLocal);

    for (int i = 0; i < nItemsLocal; ++i) {
      // Maybe use arma::cols for subview of p
      qGrad.col(i) = nRiLocal(i) * lambdaqpy * q.col(i + iOffset);

      for (const auto u : RiLocal[i]) {
        qGrad.col(i) -=
            iE(i, u) *
            (p.col(u) + isqNRu(u) * iE(i, u) * arma::sum(y.cols(Ru[u]), 1));
      }

      float prefactor = 0.0;
      for (const auto u : RiLocal[i]) {
        prefactor += isqNRu(u) * iE(i, u);
      }

      yGrad.col(i) = nRiLocal(i) * lambdaqpy * y.col(i + iOffset) -
                     prefactor * q.col(i + iOffset);
    }

    for (int u = 0; u < nUsersLocal; ++u) {
      pGrad.col(u) = nRuLocal(u) * lambdaqpy * p.col(u + uOffset);

      for (const auto i : RuLocal[u]) {
        pGrad.col(u) -= q.col(i) * uE(i, u);
      }
    }

    qGrad *= 2;
    pGrad *= 2;
    yGrad *= 2;

    // Compute update steps
    const arma::fvec biStep = -gammab * biGrad;
    const arma::fvec buStep = -gammab * buGrad;
    const arma::fmat qStep = -gammaqpy * qGrad;
    const arma::fmat pStep = -gammaqpy * pGrad;
    const arma::fmat yStep = -gammaqpy * yGrad;

    // Update tables with steps
    this->stepTable(biTable, biStep, 0, iOffset);
    this->stepTable(buTable, buStep, 0, uOffset);
    this->stepTable(qTable, qStep, iOffset, 0);
    this->stepTable(pTable, pStep, uOffset, 0);
    this->stepTable(yTable, yStep, iOffset, 0);

    petuum::PSTableGroup::GlobalBarrier();

    // Load updated parameters
    bi = gaml::util::table::loadMatrix(biTable, nItems, 1);
    bu = gaml::util::table::loadMatrix(buTable, nUsers, 1);
    q = gaml::util::table::loadMatrix(qTable, k, nItems);
    p = gaml::util::table::loadMatrix(pTable, k, nUsers);
    y = gaml::util::table::loadMatrix(yTable, k, nItems);

    // Compute the Squared Error for logging
    float se = 0.0;
    for (arma::sp_fmat::const_iterator it = nUSlice.begin(),
                                       end = nUSlice.end();
         it != end; ++it) {
      const arma::uword i = it.row();
      const arma::uword u = it.col();
      const float error =
          *it - bi(i) - bu(u) -
          arma::dot(q.col(i),
                    p.col(u + uOffset) +
                        isqNRuLocal(u) * arma::sum(y.cols(RuLocal[u]), 1));

      se += error * error;
    }

    seTable.Inc(0, this->rank, se - prevSE);
    prevSE = se;

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch all updated SEs
    const arma::fmat SEs = gaml::util::table::loadMatrix(seTable, nranks, 1);

    // Compute MSE
    const float mse = arma::sum(SEs.col(0)) / nnz;

    // Absolute and relative improvements
    aimprov = prevMSE - mse;
    rimprov = aimprov / prevMSE;
    prevMSE = mse;

    if (this->rank == 0) {
      std::cout << "Iteration " << iteration << ": MSE = " << mse
                << ", AImp = " << aimprov << ", RImp = " << rimprov
                << std::endl;
    }

    gammab *= 0.9;
    gammaqpy *= 0.9;
  } while (iteration == 1 || rimprov >= rtol && aimprov >= atol);

  return {mu, bi, bu, q, p, y};
}

void Worker::initTables(int muTableId, int biTableId, int buTableId,
                        int qTableId, int pTableId, int yTableId,
                        int errorTableId, int nruTableId, int ruTableId,
                        int seTableId, int floatRowType, int intRowType, int k,
                        int nnz, int maxFill, int nItems, int nUsers,
                        int nranks) {
  petuum::ClientTableConfig muConfig;
  muConfig.table_info.row_type = floatRowType;
  muConfig.table_info.row_capacity = nranks;
  muConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  muConfig.table_info.table_staleness = 0;
  muConfig.table_info.oplog_dense_serialized = true;
  muConfig.table_info.dense_row_oplog_capacity =
      muConfig.table_info.row_capacity;
  muConfig.process_cache_capacity = 2;
  muConfig.oplog_capacity = 2;
  muConfig.thread_cache_capacity = 1;
  muConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(muTableId, muConfig);

  petuum::ClientTableConfig biConfig;
  biConfig.table_info.row_type = floatRowType;
  biConfig.table_info.row_capacity = nItems;
  biConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  biConfig.table_info.table_staleness = 0;
  biConfig.table_info.oplog_dense_serialized = true;
  biConfig.table_info.dense_row_oplog_capacity =
      biConfig.table_info.row_capacity;
  biConfig.process_cache_capacity = 1;
  biConfig.oplog_capacity = 1;
  biConfig.thread_cache_capacity = 1;
  biConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(biTableId, biConfig);

  petuum::ClientTableConfig buConfig;
  buConfig.table_info.row_type = floatRowType;
  buConfig.table_info.row_capacity = nUsers;
  buConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  buConfig.table_info.table_staleness = 0;
  buConfig.table_info.oplog_dense_serialized = true;
  buConfig.table_info.dense_row_oplog_capacity =
      buConfig.table_info.row_capacity;
  buConfig.process_cache_capacity = 1;
  buConfig.oplog_capacity = 1;
  buConfig.thread_cache_capacity = 1;
  buConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(buTableId, buConfig);

  petuum::ClientTableConfig qConfig;
  qConfig.table_info.row_type = floatRowType;
  qConfig.table_info.row_capacity = k;
  qConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  qConfig.table_info.table_staleness = 0;
  qConfig.table_info.oplog_dense_serialized = true;
  qConfig.table_info.dense_row_oplog_capacity = qConfig.table_info.row_capacity;
  qConfig.process_cache_capacity = nItems;
  qConfig.oplog_capacity = nItems;
  qConfig.thread_cache_capacity = 1;
  qConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(qTableId, qConfig);

  petuum::ClientTableConfig pConfig;
  pConfig.table_info.row_type = floatRowType;
  pConfig.table_info.row_capacity = k;
  pConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  pConfig.table_info.table_staleness = 0;
  pConfig.table_info.oplog_dense_serialized = true;
  pConfig.table_info.dense_row_oplog_capacity = pConfig.table_info.row_capacity;
  pConfig.process_cache_capacity = nUsers;
  pConfig.oplog_capacity = nUsers;
  pConfig.thread_cache_capacity = 1;
  pConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(pTableId, pConfig);

  petuum::ClientTableConfig yConfig;
  yConfig.table_info.row_type = floatRowType;
  yConfig.table_info.row_capacity = k;
  yConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  yConfig.table_info.table_staleness = 0;
  yConfig.table_info.oplog_dense_serialized = true;
  yConfig.table_info.dense_row_oplog_capacity = yConfig.table_info.row_capacity;
  yConfig.process_cache_capacity = nUsers;
  yConfig.oplog_capacity = nUsers;
  yConfig.thread_cache_capacity = 1;
  yConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(yTableId, yConfig);

  petuum::ClientTableConfig errorConfig;
  errorConfig.table_info.row_type = floatRowType;
  errorConfig.table_info.row_capacity = nnz;
  errorConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  errorConfig.table_info.table_staleness = 0;
  errorConfig.table_info.oplog_dense_serialized = true;
  errorConfig.table_info.dense_row_oplog_capacity =
      errorConfig.table_info.row_capacity;
  errorConfig.process_cache_capacity = 1;
  errorConfig.oplog_capacity = 1;
  errorConfig.thread_cache_capacity = 1;
  errorConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(errorTableId, errorConfig);

  petuum::ClientTableConfig nruConfig;
  nruConfig.table_info.row_type = intRowType;
  nruConfig.table_info.row_capacity = nUsers;
  nruConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  nruConfig.table_info.table_staleness = 0;
  nruConfig.table_info.oplog_dense_serialized = true;
  nruConfig.table_info.dense_row_oplog_capacity =
      nruConfig.table_info.row_capacity;
  nruConfig.process_cache_capacity = 1;
  nruConfig.oplog_capacity = 1;
  nruConfig.thread_cache_capacity = 1;
  nruConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(nruTableId, nruConfig);

  petuum::ClientTableConfig ruConfig;
  ruConfig.table_info.row_type = intRowType;
  ruConfig.table_info.row_capacity = maxFill;
  ruConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  ruConfig.table_info.table_staleness = 0;
  ruConfig.table_info.oplog_dense_serialized = true;
  ruConfig.table_info.dense_row_oplog_capacity =
      ruConfig.table_info.row_capacity;
  ruConfig.process_cache_capacity = nUsers;
  ruConfig.oplog_capacity = nUsers;
  ruConfig.thread_cache_capacity = 1;
  ruConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(ruTableId, ruConfig);

  petuum::ClientTableConfig seConfig;
  seConfig.table_info.row_type = floatRowType;
  seConfig.table_info.row_capacity = nranks;
  seConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  seConfig.table_info.table_staleness = 0;
  seConfig.table_info.oplog_dense_serialized = true;
  seConfig.table_info.dense_row_oplog_capacity =
      seConfig.table_info.row_capacity;
  seConfig.process_cache_capacity = 1;
  seConfig.oplog_capacity = 1;
  seConfig.thread_cache_capacity = 1;
  seConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(seTableId, seConfig);
}

// Apply the m*n step to the table
void Worker::stepTable(petuum::Table<float>& table, const arma::fmat& step,
                       int rowOffset, int colOffset) {
  // step is an m*n table
  int m = step.n_rows;
  int n = step.n_cols;

  for (int j = 0; j < n; j++) {
    petuum::DenseUpdateBatch<float> batch(colOffset, m);

    std::memcpy(batch.get_mem(), step.colptr(j), m * sizeof(float));

    table.DenseBatchInc(j + rowOffset, batch);
  }
}
}
}
}
