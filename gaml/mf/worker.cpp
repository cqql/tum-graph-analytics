#include "worker.h"

namespace gaml {

namespace mf {

void Worker::randomizeTable(petuum::Table<float>& table, int m, int n,
                            int offset) {
  arma::fvec vec(n);

  for (int i = 0; i < m; i++) {
    vec.randn();
    vec = arma::abs(vec);

    petuum::DenseUpdateBatch<float> batch(offset, n);
    std::memcpy(batch.get_mem(), vec.memptr(), n * sizeof(float));

    table.DenseBatchInc(i, batch);
  }
}
}
}
