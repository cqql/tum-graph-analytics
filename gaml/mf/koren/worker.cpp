#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "../../util/table.h"
#include "worker.h"

namespace gaml {

namespace mf {

namespace koren {

// A sign function from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

// Subtract b from all non-zero entries of A
arma::sp_fmat operator-(const arma::sp_fmat& A, const float& b) {
  const arma::uvec rowinds(A.row_indices, A.n_nonzero);
  const arma::uvec colptrs(A.col_ptrs, A.n_cols + 1);
  arma::fvec vals(A.values, A.n_nonzero);

  vals -= b * arma::ones<arma::fvec>(A.n_nonzero);

  return arma::sp_fmat(rowinds, colptrs, vals, A.n_rows, A.n_cols);
}

std::tuple<float, arma::fvec, arma::fvec, arma::fvec, arma::fvec, arma::fmat,
           arma::fmat, arma::fmat>
Worker::factor(const arma::sp_fmat iSlice, const int iOffset,
               const arma::sp_fmat uSlice, const arma::sp_fmat tSlice,
               const int uOffset, const int k) {
  const auto rank = this->rank;
  const auto nranks = this->nranks;
  const auto lambdab = this->lambdab;
  const auto lambdaqpy = this->lambdaqpy;
  auto gammab = this->gammab;
  auto gammat = this->gammat;
  auto gammaqpy = this->gammaqpy;
  const auto beta = this->beta;
  const auto atol = this->atol;
  const auto rtol = this->rtol;
  auto tiTable = this->tiTable;
  auto muTable = this->muTable;
  auto biTable = this->biTable;
  auto buTable = this->buTable;
  auto alphaTable = this->alphaTable;
  auto kappaTable = this->kappaTable;
  auto qTable = this->qTable;
  auto yTable = this->yTable;
  auto pTable = this->pTable;
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
  //
  // This is what we would normally do and replace all usages of uSlice with
  // nUSlice. But armadillo sparse matrices filter out all zero-entries on
  // creation, so this way we would lose the entries that are perfectly
  // predicted by the mean.
  // const arma::sp_fmat nUSlice = uSlice - mu;

  // Initialize all the parameters
  arma::fvec bi(nItems, arma::fill::randn);
  arma::fvec bu(nUsers, arma::fill::randn);
  arma::fvec alpha(nUsers, arma::fill::zeros);
  arma::fvec kappa(nItems, arma::fill::zeros);
  arma::fmat q(k, nItems, arma::fill::randn);
  arma::fmat p(k, nUsers, arma::fill::randn);
  arma::fmat y(k, nItems, arma::fill::randn);

  // NOTE: We do not store the parameters in the table initially. This means
  // that each worker uses a different set of parameters in the first iteration.
  // However, an experiment has shown that this does not hurt learning at all
  // and thus is purely beneficial for performance.

  // Mean time of rating per user
  arma::fvec tU(nUsersLocal);
  for (int i = 0; i < nUsersLocal; ++i) {
    // Number of non-zero entries in this column
    const int nnz = tSlice.col_ptrs[i + 1] - tSlice.col_ptrs[i];

    tU[i] = arma::mean(arma::fvec(&tSlice.values[tSlice.col_ptrs[i]], nnz));
  }

  // Mean time of rating per item
  arma::fmat tiStats;
  {
    // Number of ratings and the sum of their dates per item in this u slice
    arma::fvec nI(nItems);
    arma::fvec tI(nItems);

    for (arma::sp_fmat::const_iterator it = tSlice.begin(), end = tSlice.end();
         it != end; ++it) {
      const int i = it.row();

      nI(i) += 1;
      tI(i) += *it;
    }

    this->stepTable(tiTable, tI, 0, 0);
    this->stepTable(tiTable, nI, 1, 0);

    petuum::PSTableGroup::GlobalBarrier();

    tiStats = gaml::util::table::loadMatrix(tiTable, nItems, 2);
  }
  const arma::fvec tI = tiStats.col(0) / tiStats.col(1);

  // Indices of all items rated by u for each user u in the local U slice
  std::vector<arma::uvec> RuLocal(nUsersLocal);
  // Number of items rated by each u
  arma::uvec nRuLocal(nUsersLocal);

  for (int i = 0; i < nUsersLocal; ++i) {
    // Number of non-zero entries in this column
    const int nnz = uSlice.col_ptrs[i + 1] - uSlice.col_ptrs[i];

    RuLocal[i] = arma::uvec(&uSlice.row_indices[uSlice.col_ptrs[i]], nnz);
    nRuLocal(i) = nnz;
  }

  // And their inverse square roots (i.e. ^-1/2)
  const arma::uvec isqNRuLocal = 1 / arma::sqrt(nRuLocal);

  // Finally the same for all items in the local U slice
  std::vector<arma::uvec> RiLocal(nItems);
  // Number of local users that rated item i
  arma::uvec nRiLocal(nItems);

  // Transpose the U slice so that we can easily compute this
  const arma::sp_fmat uSliceT = uSlice.t();
  for (int i = 0; i < nItems; ++i) {
    // Number of non-zero entries in this column
    const int nnz = uSliceT.col_ptrs[i + 1] - uSliceT.col_ptrs[i];

    RiLocal[i] = arma::uvec(&uSliceT.row_indices[uSliceT.col_ptrs[i]], nnz);
    nRiLocal(i) = nnz;
  }

  // Cache the locations of entries in the local U slice
  arma::umat uLocations(2, uSlice.n_nonzero);
  {
    int index = 0;
    for (arma::sp_fmat::const_iterator it = uSlice.begin(), end = uSlice.end();
         it != end; ++it, ++index) {
      uLocations(0, index) = it.row();
      uLocations(1, index) = it.col();
    }
  }

  // Time deviations of each rating
  arma::fvec uDevValues(uSlice.n_nonzero);
  arma::fvec iDevValues(uSlice.n_nonzero);
  {
    int index = 0;
    for (arma::sp_fmat::const_iterator it = tSlice.begin(), end = tSlice.end();
         it != end; ++it, ++index) {
      const int i = it.row();
      const int u = it.col();
      const float tuDev = *it - tU(u);
      const float tiDev = *it - tI(i);

      uDevValues(index) = sgn(tuDev) * std::pow(std::abs(tuDev), beta);
      iDevValues(index) = sgn(tiDev) * std::pow(std::abs(tiDev), beta);
    }
  }
  arma::sp_fmat uDev(uLocations, uDevValues, uSlice.n_rows, uSlice.n_cols);
  arma::sp_fmat iDev(uLocations, iDevValues, uSlice.n_rows, uSlice.n_cols);

  // Values from previous iteration to compute the diff against for updates
  float prevSE = 0.0;
  float prevMSE = 0.0;
  float aimprov = 0.0;
  float rimprov = 0.0;

  // Run the gradient descent
  int iteration = 0;
  do {
    iteration++;

    // Prediction error matrix for the U slice
    arma::fvec uErrors(uSlice.n_nonzero);
    {
      int index = 0;
      for (arma::sp_fmat::const_iterator it = uSlice.begin(),
                                         end = uSlice.end();
           it != end; ++it, ++index) {
        const arma::uword i = it.row();
        const arma::uword u = it.col();
        const arma::uword uGlobal = u + uOffset;

        uErrors(index) =
            *it - mu - bi(i) - bu(uGlobal) - alpha(uGlobal) * uDev(i, u) -
            kappa(i) * iDev(i, u) -
            arma::dot(q.col(i),
                      p.col(uGlobal) +
                          isqNRuLocal(u) * arma::sum(y.cols(RuLocal[u]), 1));
      }
    }
    const arma::sp_fmat uE(uLocations, uErrors, uSlice.n_rows, uSlice.n_cols);

    // Compute (partial) gradients
    const arma::fvec biGrad = 2 * (lambdab * nRiLocal % bi - arma::sum(uE, 1));
    const arma::fvec buGrad =
        2 * (lambdab * nRuLocal % bu.rows(uOffset, uOffset + nUsersLocal - 1) -
             arma::sum(uE, 0).t());
    const arma::fvec alphaGrad =
        2 *
        (lambdab * nRuLocal % alpha.rows(uOffset, uOffset + nUsersLocal - 1) -
         arma::sum(uDev % uE, 0).t());
    const arma::fvec kappaGrad =
        2 * (lambdab * nRiLocal % kappa - arma::sum(iDev % uE, 1));

    arma::fmat qGrad(k, nItems);
    arma::fmat pGrad(k, nUsersLocal);
    arma::fmat yGrad(k, nItems);

    for (int i = 0; i < nItems; ++i) {
      // Maybe use arma::cols for subview of p
      qGrad.col(i) = nRiLocal(i) * lambdaqpy * q.col(i);

      for (const auto u : RiLocal[i]) {
        qGrad.col(i) -=
            uE(i, u) * (p.col(u + uOffset) +
                        isqNRuLocal(u) * arma::sum(y.cols(RuLocal[u]), 1));
      }

      float prefactor = 0.0;
      for (const auto u : RiLocal[i]) {
        prefactor += isqNRuLocal(u) * uE(i, u);
      }

      yGrad.col(i) = nRiLocal(i) * lambdaqpy * y.col(i) - prefactor * q.col(i);
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
    const arma::fvec alphaStep = -gammat * alphaGrad;
    const arma::fvec kappaStep = -(gammat / 5) * kappaGrad;
    const arma::fmat qStep = -gammaqpy * qGrad;
    const arma::fmat pStep = -gammaqpy * pGrad;
    const arma::fmat yStep = -gammaqpy * yGrad;

    // Update tables with steps
    this->stepTable(biTable, biStep, 0, 0);
    this->stepTable(buTable, buStep, 0, uOffset);
    this->stepTable(alphaTable, alphaStep, 0, uOffset);
    this->stepTable(kappaTable, kappaStep, 0, 0);
    this->stepTable(qTable, qStep.t(), 0, 0);
    this->stepTable(pTable, pStep.t(), 0, uOffset);
    this->stepTable(yTable, yStep.t(), 0, 0);

    petuum::PSTableGroup::GlobalBarrier();

    // Load updated parameters
    bi = gaml::util::table::loadMatrix(biTable, nItems, 1);
    bu = gaml::util::table::loadMatrix(buTable, nUsers, 1);
    alpha = gaml::util::table::loadMatrix(alphaTable, nUsers, 1);
    kappa = gaml::util::table::loadMatrix(kappaTable, nItems, 1);
    q = gaml::util::table::loadMatrix(qTable, nItems, k).t();
    p = gaml::util::table::loadMatrix(pTable, nUsers, k).t();
    y = gaml::util::table::loadMatrix(yTable, nItems, k).t();

    // This one is applied directly because it is used purely locally

    // Compute the Squared Error for logging
    float se = 0.0;
    for (arma::sp_fmat::const_iterator it = uSlice.begin(), end = uSlice.end();
         it != end; ++it) {
      const arma::uword i = it.row();
      const arma::uword u = it.col();
      const arma::uword uGlobal = u + uOffset;
      const float error =
          *it - mu - bi(i) - bu(uGlobal) - alpha(uGlobal) * uDev(i, u) -
          kappa(i) * iDev(i, u) -
          arma::dot(q.col(i),
                    p.col(uGlobal) +
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

    // This actually just hurts performance but does not help convergence for
    // our datasets
    // gammab *= 0.9;
    // gammaqpy *= 0.9;
  } while (iteration == 1 || rimprov >= rtol && aimprov >= atol);

  return std::make_tuple(mu, bi, bu, alpha, kappa, q, p, y);
}

void Worker::initTables(int tiTableId, int muTableId, int biTableId,
                        int buTableId, int alphaTableId, int kappaTableId,
                        int qTableId, int pTableId, int yTableId, int seTableId,
                        int floatRowType, int intRowType, int k, int nItems,
                        int nUsers, int nranks) {
  petuum::ClientTableConfig tiConfig;
  tiConfig.table_info.row_type = floatRowType;
  tiConfig.table_info.row_capacity = nItems;
  tiConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  tiConfig.table_info.table_staleness = 0;
  tiConfig.table_info.oplog_dense_serialized = true;
  tiConfig.table_info.dense_row_oplog_capacity =
      tiConfig.table_info.row_capacity;
  tiConfig.process_cache_capacity = 2;
  tiConfig.oplog_capacity = 2;
  tiConfig.thread_cache_capacity = 1;
  tiConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(tiTableId, tiConfig);

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

  petuum::ClientTableConfig alphaConfig;
  alphaConfig.table_info.row_type = floatRowType;
  alphaConfig.table_info.row_capacity = nUsers;
  alphaConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  alphaConfig.table_info.table_staleness = 0;
  alphaConfig.table_info.oplog_dense_serialized = true;
  alphaConfig.table_info.dense_row_oplog_capacity =
      alphaConfig.table_info.row_capacity;
  alphaConfig.process_cache_capacity = 1;
  alphaConfig.oplog_capacity = 1;
  alphaConfig.thread_cache_capacity = 1;
  alphaConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(alphaTableId, alphaConfig);

  petuum::ClientTableConfig kappaConfig;
  kappaConfig.table_info.row_type = floatRowType;
  kappaConfig.table_info.row_capacity = nItems;
  kappaConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  kappaConfig.table_info.table_staleness = 0;
  kappaConfig.table_info.oplog_dense_serialized = true;
  kappaConfig.table_info.dense_row_oplog_capacity =
      kappaConfig.table_info.row_capacity;
  kappaConfig.process_cache_capacity = 1;
  kappaConfig.oplog_capacity = 1;
  kappaConfig.thread_cache_capacity = 1;
  kappaConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(kappaTableId, kappaConfig);

  petuum::ClientTableConfig qConfig;
  qConfig.table_info.row_type = floatRowType;
  qConfig.table_info.row_capacity = nItems;
  qConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  qConfig.table_info.table_staleness = 0;
  qConfig.table_info.oplog_dense_serialized = true;
  qConfig.table_info.dense_row_oplog_capacity = qConfig.table_info.row_capacity;
  qConfig.process_cache_capacity = k;
  qConfig.oplog_capacity = k;
  qConfig.thread_cache_capacity = 1;
  qConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(qTableId, qConfig);

  petuum::ClientTableConfig pConfig;
  pConfig.table_info.row_type = floatRowType;
  pConfig.table_info.row_capacity = nUsers;
  pConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  pConfig.table_info.table_staleness = 0;
  pConfig.table_info.oplog_dense_serialized = true;
  pConfig.table_info.dense_row_oplog_capacity = pConfig.table_info.row_capacity;
  pConfig.process_cache_capacity = k;
  pConfig.oplog_capacity = k;
  pConfig.thread_cache_capacity = 1;
  pConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(pTableId, pConfig);

  petuum::ClientTableConfig yConfig;
  yConfig.table_info.row_type = floatRowType;
  yConfig.table_info.row_capacity = nItems;
  yConfig.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  yConfig.table_info.table_staleness = 0;
  yConfig.table_info.oplog_dense_serialized = true;
  yConfig.table_info.dense_row_oplog_capacity = yConfig.table_info.row_capacity;
  yConfig.process_cache_capacity = k;
  yConfig.oplog_capacity = k;
  yConfig.thread_cache_capacity = 1;
  yConfig.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(yTableId, yConfig);

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
