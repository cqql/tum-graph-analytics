#include <fstream>
//#include <iomanip>
#include <fenv.h>

#include <armadillo>

#include "worker.hpp"

#include "../io/tensor.hpp"
#include "../util/table.h"

namespace gaml {
namespace tf {

static const int mse_log = 10;

Worker::Worker(int id, int rank, int iterations, int usertableid, int prodtableid, int wordtableid, int setableid,
               int useroffset, int prodoffset, int wordoffset, 
               const gaml::io::Sparse3dTensor& Ruser, 
               const gaml::io::Sparse3dTensor& Rprod, 
               const gaml::io::Sparse3dTensor& Rword, 
               const gaml::io::Sparse3dTensor& Rvali,
               const gaml::io::Sparse3dTensor& Rtest)
    : id(id),
      rank(rank),
      iterations(iterations),
      usertable(petuum::PSTableGroup::GetTableOrDie<float>(usertableid)),
      prodtable(petuum::PSTableGroup::GetTableOrDie<float>(prodtableid)),
      wordtable(petuum::PSTableGroup::GetTableOrDie<float>(wordtableid)),
      setable(petuum::PSTableGroup::GetTableOrDie<float>(setableid)),
      useroffset(useroffset),
      prodoffset(prodoffset),
      wordoffset(wordoffset),
      Ruser(Ruser),
      Rprod(Rprod),
      Rword(Rword),
      Rvali(Rvali),
      Rtest(Rtest),
      se_train_vec(std::vector<float>(mse_log)),
      se_vali_vec(std::vector<float>(mse_log)) {}

std::tuple<arma::fmat, arma::fmat, arma::fmat> Worker::factorize(float lambda, bool clamp, bool reg, int reg_thr, int stop_tol) {

  feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
  petuum::RowAccessor rowacc;
  
  // Initialize tables with random values
  //arma::arma_rng::set_seed_random();
  gaml::util::table::randomizeTable(usertable, rank, Ruser.n_rows, useroffset);
  gaml::util::table::randomizeTable(prodtable, rank, Rprod.n_cols, prodoffset);
  gaml::util::table::randomizeTable(wordtable, rank, Rword.n_words, wordoffset);
  
  float last_se_train=0;
  float last_se_vali=0;
  setable.Inc(1, id*2, Rvali.n_nz);
  setable.Inc(1, id*2+1, Rtest.n_nz);
  
  petuum::PSTableGroup::GlobalBarrier();
  
  // Fetch U, P and T
  auto U = gaml::util::table::loadMatrix(usertable, Rword.n_rows, rank);
  auto P = gaml::util::table::loadMatrix(prodtable, Rword.n_cols, rank);
  auto T = gaml::util::table::loadMatrix(wordtable, Ruser.n_words, rank);
  
  auto sum_sizes = read_split_sum(1);
  auto vali_size = std::get<0>(sum_sizes);
  auto test_size = std::get<1>(sum_sizes);
  
  for (int round = 0; round < iterations; round++) {
    ///////
    // Compute gradient for U
    ///////
    arma::fmat Ugrad(Ruser.n_rows, rank, arma::fill::zeros);
    arma::fmat Unum(Ruser.n_rows, rank, arma::fill::zeros);
    arma::fmat Udenom(Ruser.n_rows, rank, arma::fill::zeros);
    
    // iterate over all up pairs in Ruser
    for (std::size_t i = 0; i != Ruser.n_nz; ++i) {
      int userind = Ruser.rows[i];
      int prodind = Ruser.cols[i];
      auto wordbag = Ruser.getWordBagAt(i);
      
      Unum.row(userind - useroffset) += P.row(prodind) % (wordbag * T);
      Udenom.row(userind - useroffset) += P.row(prodind) % ((U.row(userind) % P.row(prodind) * T.t()) * T);  
    }
    
    arma::fmat Ulocal = U.rows(useroffset, useroffset + Ruser.n_rows - 1);
    // prevent div by zero
    Udenom += 10E-16f;
    Ugrad = (Ulocal % Unum / Udenom) - Ulocal;
    if(reg && round > reg_thr) {
      Ugrad = Ugrad - lambda * Ulocal % Ulocal / Udenom;
    }

    // Update U table
    gaml::util::table::updateMatrixSlice(Ugrad, usertable, Ugrad.n_rows, Ugrad.n_cols, useroffset);

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated U
    U = gaml::util::table::loadMatrix(usertable, U.n_rows, U.n_cols);
    if(clamp){
      U = arma::clamp(U, 0.0, std::numeric_limits<float>::max());
    }

    ///////
    // Compute gradient for P
    ///////
    arma::fmat Pgrad(Rprod.n_cols, rank, arma::fill::zeros);
    arma::fmat Pnum(Rprod.n_cols, rank, arma::fill::zeros);
    arma::fmat Pdenom(Rprod.n_cols, rank, arma::fill::zeros);
    
    // iterate over all up pairs in Rprod
    for (std::size_t i = 0; i != Rprod.n_nz; ++i) {
      int userind = Rprod.rows[i];
      int prodind = Rprod.cols[i];
      auto wordbag = Rprod.getWordBagAt(i);
      
      Pnum.row(prodind - prodoffset) += U.row(userind) % (wordbag * T);
      Pdenom.row(prodind - prodoffset) += U.row(userind) % ((U.row(userind) % P.row(prodind) * T.t()) * T);
    }
    
    arma::fmat Plocal = P.rows(prodoffset, prodoffset + Rprod.n_cols   - 1);
    Pdenom += 10E-16f;
    Pgrad = (Plocal % Pnum / Pdenom) - Plocal;
    if(reg && round > reg_thr) {
      Pgrad = Pgrad - lambda * Plocal % Plocal / Pdenom;
    }

    // Update P table
    gaml::util::table::updateMatrixSlice(Pgrad, prodtable, Pgrad.n_rows, Pgrad.n_cols, prodoffset);

    petuum::PSTableGroup::GlobalBarrier();
  
    // Fetch updated P
    P = gaml::util::table::loadMatrix(prodtable, P.n_rows, P.n_cols);
    if(clamp) {
      P = arma::clamp(P, 0.0, std::numeric_limits<float>::max());
    }

    ///////
    // Compute gradient for T
    ///////
    arma::fmat Tgrad(Rword.n_words, rank, arma::fill::zeros);
    arma::fmat Tnum(Rword.n_words, rank, arma::fill::zeros);
    arma::fmat Tdenom(Rword.n_words, rank, arma::fill::zeros);
    arma::fmat Tlocal = T.rows(wordoffset, Rword.n_words + wordoffset - 1);
    
    // iterate over all uv pairs in Rword
    for (std::size_t i = 0; i != Rword.n_nz; ++i) {
      int userind = Rword.rows[i];
      int prodind = Rword.cols[i];
      
      auto wordbag = Rword.getWordBagAt(i);
      arma::frowvec user_times_prod = (U.row(userind) % P.row(prodind));
      arma::frowvec pred = user_times_prod * Tlocal.t();

      Tnum += wordbag.t() * user_times_prod;
      Tdenom += pred.t() * user_times_prod;
    }
    Tdenom += 10E-16f;
    Tgrad = (Tlocal % Tnum / Tdenom) - Tlocal;
    if(reg && round > reg_thr) {
      Tgrad = Tgrad - lambda * Tlocal % Tlocal / Tdenom;
    }

    // Update T table
    gaml::util::table::updateMatrixSlice(Tgrad, wordtable, Tgrad.n_rows, Tgrad.n_cols, wordoffset);

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated T
    T = gaml::util::table::loadMatrix(wordtable, T.n_rows, T.n_cols);
    if(clamp) {
      T = arma::clamp(T, 0.0, std::numeric_limits<float>::max());
    }
    
    update_setable(U, P, T, round, last_se_train, last_se_vali);
    
    petuum::PSTableGroup::GlobalBarrier();
    
    update_mse(round);
    
    if(id == 0) {
      output(round+1, se_train_vec[round % mse_log] / Rword.n_nz, se_vali_vec[round % mse_log] / vali_size);
    }
    if(check_stop(round, stop_tol)) {
      break;
    }
  }
  
  float se_test = eval(U, P, T, Rtest);
  setable.Inc(2, id, se_test);  
  
  petuum::PSTableGroup::GlobalBarrier();
  
  if(id == 0) {
    std::vector<float> se_test_vec;
    const auto& row = setable.Get<petuum::DenseRow<float>>(2, &rowacc);
    row.CopyToVector(&se_test_vec);
    float mse_test = std::accumulate(se_test_vec.begin(), se_test_vec.end(), 0.0f);
    std::cout << "MSE test: " << mse_test / test_size << std::endl;
  }
  return std::make_tuple(U, P, T);
}

void Worker::output(int round, float se_train_vec, float se_vali_vec) {
  std::cout << "Round: " << round << " MSE train: " << se_train_vec << " MSE validation: " << se_vali_vec << std::endl;
}

float Worker::eval(arma::fmat& U, arma::fmat& P, arma::fmat& T, const gaml::io::Sparse3dTensor& R) {
  float se = 0;

  for (size_t i = 0; i != R.n_nz; ++i) {
    int userind = R.rows[i];
    int prodind = R.cols[i];

    arma::frowvec error =
        R.getWordBagAt(i) - (U.row(userind) % P.row(prodind)) * T.t();
    se += arma::dot(error, error);
  }

  return se;
}

void Worker::update_setable(arma::fmat& U, arma::fmat& P, arma::fmat& T, int round, float& last_se_train, float& last_se_vali) {
  float se_train = eval(U, P, T, Ruser);
  float se_vali = eval(U, P, T, Rvali);
  float tmp = last_se_train;
  se_train = (last_se_train = se_train) - tmp;
  tmp = last_se_vali;
  se_vali = (last_se_vali = se_vali) - tmp;
  
  setable.Inc(0, id*2, se_train);
  setable.Inc(0, id*2+1, se_vali);
}

void Worker::update_mse(int round) {
  auto se = read_split_sum(0);
  se_train_vec[round % mse_log] = std::get<0>(se);
  se_vali_vec[round % mse_log] = std::get<1>(se);
}

std::tuple<float, float> Worker::read_split_sum(int type) {
  petuum::RowAccessor rowacc;
  std::vector<float> even;
  std::vector<float> odd;
  std::vector<float> buf;
  bool toggle = false;
  const auto& row = setable.Get<petuum::DenseRow<float>>(type, &rowacc);
  row.CopyToVector(&buf);
  std::partition_copy(buf.begin(), buf.end(), std::back_inserter(even), std::back_inserter(odd), [&toggle](float){ return toggle = !toggle;});
  return std::make_tuple(std::accumulate(even.begin(), even.end(), 0.0f), std::accumulate(odd.begin(), odd.end(), 0.0f));
}

bool Worker::check_stop(int round, int stop_tol) {
  if(round < stop_tol) {
    return false;
  }
  for(int i = 0; i < stop_tol; i++) {
    if(se_vali_vec[(round - i) % mse_log] < se_vali_vec[(round - i - 1) % mse_log]){
      return false;
    }
  }
  return true;
}


void Worker::initTables(int uTableId, int pTableId, int tTableId, int seTableId, int rowType, int k,
                        int uNumRows, int pNumRows, int tNumRows, int num_eval, int num_workers) {
  // Create tables
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

  petuum::ClientTableConfig t_config;
  t_config.table_info.row_type = rowType;
  t_config.table_info.row_capacity = tNumRows;
  t_config.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  t_config.table_info.table_staleness = 0;
  t_config.table_info.oplog_dense_serialized = true;
  t_config.table_info.dense_row_oplog_capacity =
      t_config.table_info.row_capacity;
  t_config.process_cache_capacity = k;
  t_config.oplog_capacity = k;
  t_config.thread_cache_capacity = 1;
  t_config.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(tTableId, t_config); 
  
  petuum::ClientTableConfig se_config;
  se_config.table_info.row_type = rowType;
  se_config.table_info.row_capacity = num_workers*2;
  se_config.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  se_config.table_info.table_staleness = 0;
  se_config.table_info.oplog_dense_serialized = true;
  se_config.table_info.dense_row_oplog_capacity =
      se_config.table_info.row_capacity;
  se_config.process_cache_capacity = 3;
  se_config.oplog_capacity = 3;
  se_config.thread_cache_capacity = 1;
  se_config.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable(seTableId, se_config); 
                      
}

} // end tf
} // end gaml
