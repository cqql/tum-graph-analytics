#ifndef WORKER_H_
#define WORKER_H_

#include <string>

#include <armadillo>
#include <petuum_ps_common/include/petuum_ps.hpp>

namespace mfals {

class Worker {
 public:
  Worker(int id, std::string basepath, int k, int iterations, int evalrounds,
         int ptableid, int utableid);

  void run();

 private:
  int id;
  std::string basepath;
  int k;
  int iterations;
  int evalrounds;
  int ptableid;
  int utableid;

  // Initialize table as an m*n matrix with random entries
  void randomizetable(petuum::Table<float>& table, int m, int n);

  // Load matrix from a table
  arma::fmat loadmat(petuum::Table<float>& table, int m, int n);

  void evaltest(arma::fmat& P, arma::fmat& UT);
  void eval(arma::fmat& P, arma::fmat& UT, arma::sp_fmat& R, int rowoffset,
            int coloffset);
};
}

#endif  // WORKER_H_
