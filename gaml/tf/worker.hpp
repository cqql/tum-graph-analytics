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
               const gaml::io::Sparse3dTensor& Rtest=gaml::io::Sparse3dTensor());

  std::tuple<arma::fmat, arma::fmat, arma::fmat> factorize(float lambda=0.5, bool clamp=false, bool reg=false, int reg_thr=1);
  
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
  const gaml::io::Sparse3dTensor& Rtest;

  // Initialize table as an m*n matrix with random entries
  void randomizetable(petuum::Table<float>& table, int m, int n, int offset);

  float eval(arma::fmat& U, arma::fmat& P, arma::fmat& T, const gaml::io::Sparse3dTensor& R);
  void output(int round, float mse_test, float mse_train);
};
} // end tf
} // end gaml

#endif  // GAML_TF_WORKER_HPP_

