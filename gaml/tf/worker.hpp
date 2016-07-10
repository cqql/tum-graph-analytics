#ifndef GAML_TF_WORKER_HPP_
#define GAML_TF_WORKER_HPP_

#include <string>

#include <armadillo>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include "../io/tensor.hpp"


namespace gaml {
namespace tf {

class Worker {
 public:
  Worker(int id, int rank, int iterations, int usertableid, int prodtableid, int wordtableid, int setableid,
               int useroffset, int prodoffset, int wordoffset, 
               const gaml::io::Sparse3dTensor& Ruser, 
               const gaml::io::Sparse3dTensor& Rprod, 
               const gaml::io::Sparse3dTensor& Rword, 
               const gaml::io::Sparse3dTensor& Rvali,
               const gaml::io::Sparse3dTensor& Rtest);

  std::tuple<arma::fmat, arma::fmat, arma::fmat> factorize(float lambda, bool clamp, bool reg, int reg_thr, int stop_tol);
  
  static void initTables(int uTableId, int pTableId, int tTableId, int seTableId, int rowType, int k,
                        int uNumRows, int pNumRows, int tNumRows, int num_eval, int num_workers);

 private:
  int id;
  int rank;
  int iterations;
  petuum::Table<float> usertable;
  petuum::Table<float> prodtable;
  petuum::Table<float> wordtable;
  petuum::Table<float> setable;
  int useroffset;
  int prodoffset;
  int wordoffset;
  const gaml::io::Sparse3dTensor& Ruser;
  const gaml::io::Sparse3dTensor& Rprod;
  const gaml::io::Sparse3dTensor& Rword;
  const gaml::io::Sparse3dTensor& Rvali;
  const gaml::io::Sparse3dTensor& Rtest;
  std::vector<float> se_train_vec;
  std::vector<float> se_vali_vec;
  
  float eval(arma::fmat& U, arma::fmat& P, arma::fmat& T, const gaml::io::Sparse3dTensor& R);
  
  void update_setable(arma::fmat& U, arma::fmat& P, arma::fmat& T, int round, float& last_se_train, float& last_se_vali);
  
  void update_mse(int round);
  
  std::tuple<float, float> read_split_sum(int row);
  
  bool check_stop(int round, int stop_tol);
  
  void output(int round, float mse_test, float mse_train);
};
} // end tf
} // end gaml

#endif  // GAML_TF_WORKER_HPP_

