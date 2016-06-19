#ifndef WORKER_H_
#define WORKER_H_

#include <string>

#include <armadillo>
#include <petuum_ps_common/include/petuum_ps.hpp>
#include "tensor_data.hpp"


namespace tfals {

class Worker {
 public:
  Worker(int id, std::string basepath, int rank, int iterations,
               int evalrounds, int usertableid, int prodtableid, int wordtableid);

  void run();

 private:
  int id;
  std::string basepath;
  int rank;
  int iterations;
  int evalrounds;
  int usertableid;
  int prodtableid;
  int wordtableid;

  // Initialize table as an m*n matrix with random entries
  void randomizetable(petuum::Table<float>& table, int m, int n);
  
  void updatetable(petuum::Table<float>& table, arma::fmat& grad, int offset);

  // Load matrix from a table
  arma::fmat loadmat(petuum::Table<float>& table, int m, int n);

  void evaltest(arma::fmat& U, arma::fmat& P, arma::fmat& T);
  void eval(arma::fmat& U, arma::fmat& P, arma::fmat& T, Sparse3dTensor& R,
            int useroffset, int prodoffset);
};
}

#endif  // WORKER_H_

