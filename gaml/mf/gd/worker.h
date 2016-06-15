#ifndef GAML_MF_GD_WORKER_H_
#define GAML_MF_GD_WORKER_H_

#include <random>
#include <vector>

#include "../worker.h"
#include "projection.h"

namespace gaml {

namespace mf {

namespace gd {

/**
 * Bosen worker for matrix factorization
 */
class Worker : public gaml::mf::Worker {
 public:
  Worker(int pTableId, int uTableId, int iterations, int minibatch,
         std::mt19937 rng, const Projection& projection)
      : gaml::mf::Worker(pTableId, uTableId),
        iterations(iterations),
        minibatch(minibatch),
        rng(rng),
        projection(projection) {}

  std::tuple<arma::fmat, arma::fmat> factor(const arma::sp_fmat pSlice,
                                            const int pOffset,
                                            const arma::sp_fmat uSlice,
                                            const int uOffset, const int k);

  static void initTables(int pTableId, int uTableId, int rowType, int k,
                         int pNumRows, int uNumRows);

 private:
  const int iterations;
  const int minibatch;
  const Projection& projection;
  std::mt19937 rng;

  /**
   * Select sorted subset indices into elements of M of size mbsize
   */
  std::vector<int> selectMinibatch(const arma::sp_fmat& M, const int mbSize);
};
}
}
}

#endif  // GAML_MF_GD_WORKER_H_
