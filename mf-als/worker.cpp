#include <fstream>
#include <iomanip>

#include <armadillo>
#include <glog/logging.h>

#include "worker.h"

#include "matrix_data.h"

namespace mfals {

Worker::Worker(int id, std::string basepath, int k, int iterations,
               int ptableid, int utableid)
    : id(id),
      basepath(basepath),
      k(k),
      iterations(iterations),
      ptableid(ptableid),
      utableid(utableid) {}

void Worker::run() {
  petuum::PSTableGroup::RegisterThread();

  std::ostringstream path;
  path << this->basepath << "/rank-" << this->id;
  struct MatrixData md = MatrixData::parse(path.str());
  int poffset = md.prodoffset;
  int uoffset = md.useroffset;
  auto Rprod = md.Rprod;
  auto Ruser = md.Ruser;

  petuum::Table<float> Ptable =
      petuum::PSTableGroup::GetTableOrDie<float>(this->ptableid);
  petuum::Table<float> Utable =
      petuum::PSTableGroup::GetTableOrDie<float>(this->utableid);

  // Register rows
  if (this->id == 0) {
    for (int i = 0; i < this->k; i++) {
      Ptable.GetAsyncForced(i);
      Utable.GetAsyncForced(i);
    }
  }

  petuum::PSTableGroup::GlobalBarrier();

  LOG(INFO) << "Randomize P and U";

  if (this->id == 0) {
    this->randomizetable(Ptable, this->k, Ruser.n_rows);
    this->randomizetable(Utable, this->k, Rprod.n_cols);
  }

  petuum::PSTableGroup::GlobalBarrier();

  LOG(INFO) << "Fetch P and U on worker " << this->id;

  // Fetch P and U^T
  auto P = this->loadmat(Ptable, Ruser.n_rows, this->k);
  auto UT = this->loadmat(Utable, Rprod.n_cols, this->k);

  LOG(INFO) << "Start optimization";

  float step = 1/256;

  for (int i = 0; i < this->iterations; i++) {
    LOG(INFO) << "Optimization round " << i << " on worker " << this->id;

    // Compute gradient for P
    arma::fmat Pgrad(Rprod.n_rows, this->k, arma::fill::zeros);

    arma::sp_fmat::const_iterator start = Rprod.begin();
    arma::sp_fmat::const_iterator end = Rprod.end();
    for (arma::sp_fmat::const_iterator it = start; it != end; ++it) {
      int row = it.row();
      int col = it.col();

      float diff = arma::dot(P.row(row + poffset), UT.row(col)) - (*it);

      for (int x = 0; x < this->k; x++) {
        Pgrad(row, x) += 2 * UT(col, x) * diff;
      }
    }

    Pgrad = Pgrad * (-step);

    // Update P table
    for (int j = 0; j < Pgrad.n_cols; j++) {
      petuum::DenseUpdateBatch<float> batch(poffset, Pgrad.n_rows);

      std::memcpy(batch.get_mem(), Pgrad.colptr(j),
                  Pgrad.n_rows * sizeof(float));

      Ptable.DenseBatchInc(j, batch);
    }

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated P
    P = this->loadmat(Ptable, Ruser.n_rows, this->k);

    // Compute gradient for U^T
    arma::fmat UTgrad(Ruser.n_cols, this->k, arma::fill::zeros);

    start = Ruser.begin();
    end = Ruser.end();
    for (arma::sp_fmat::const_iterator it = start; it != end; ++it) {
      int row = it.row();
      int col = it.col();

      float diff = arma::dot(P.row(row), UT.row(col + uoffset)) - (*it);

      for (int x = 0; x < this->k; x++) {
        UTgrad(col, x) += 2 * P(row, x) * diff;
      }
    }

    UTgrad = UTgrad * (-step);

    // Update U table
    for (int j = 0; j < UTgrad.n_cols; j++) {
      petuum::DenseUpdateBatch<float> batch(uoffset, UTgrad.n_rows);

      std::memcpy(batch.get_mem(), UTgrad.colptr(j),
                  UTgrad.n_rows * sizeof(float));

      Utable.DenseBatchInc(j, batch);
    }

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated U^T
    UT = this->loadmat(Utable, Rprod.n_cols, this->k);
  }

  // Evaluate on test data
  if (this->id == 0) {
    std::ostringstream testpath;
    testpath << this->basepath << "/test";
    std::ifstream f(testpath.str(), std::ios::binary);
    auto Rtest = MatrixData::parsemat(f);

    float mse = 0;

    arma::sp_fmat::const_iterator start = Rtest.begin();
    arma::sp_fmat::const_iterator end = Rtest.end();
    for (arma::sp_fmat::const_iterator it = start; it != end; ++it) {
      int row = it.row();
      int col = it.col();

      float error = (*it) - arma::dot(P.row(row), UT.row(col));
      mse += error * error;

      LOG(INFO) << "Product " << std::setw(7) << row << ", User "
                << std::setw(7) << col << ": " << std::setw(7) << error << " (" << *it << ")";
    }

    LOG(INFO) << "MSE = " << mse / Rtest.n_elem;
  }

  LOG(INFO) << "Shutdown worker " << this->id;

  petuum::PSTableGroup::DeregisterThread();
}

// Initialize table as an m*n matrix with random entries
void Worker::randomizetable(petuum::Table<float>& table, int m, int n) {
  arma::fvec vec(n);

  for (int i = 0; i < m; i++) {
    vec.randn();

    petuum::DenseUpdateBatch<float> batch(0, n);
    std::memcpy(batch.get_mem(), vec.memptr(), n * sizeof(float));

    table.DenseBatchInc(i, batch);
  }
}

arma::fmat Worker::loadmat(petuum::Table<float>& table, int m, int n) {
  arma::fmat M(m, n);
  petuum::RowAccessor rowacc;

  for (int i = 0; i < n; i++) {
    std::vector<float> tmp;
    const auto& col = table.Get<petuum::DenseRow<float>>(i, &rowacc);
    col.CopyToVector(&tmp);
    std::memcpy(M.colptr(i), tmp.data(), sizeof(float) * m);
  }

  return M;
}
}
