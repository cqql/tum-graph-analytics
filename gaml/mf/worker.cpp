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

arma::fmat Worker::loadMatrix(petuum::Table<float>& table, int m, int n) {
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
}
