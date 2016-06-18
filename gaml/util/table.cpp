#include "table.h"

namespace gaml {

namespace util {

namespace table {

arma::fmat loadMatrix(petuum::Table<float>& table, int m, int n) {
  arma::fmat M(m, n);
  std::vector<float> buf(m);
  petuum::RowAccessor rowacc;

  for (int i = 0; i < n; i++) {
    const auto& col = table.Get<petuum::DenseRow<float>>(i, &rowacc);
    col.CopyToVector(&buf);
    std::memcpy(M.colptr(i), buf.data(), m * sizeof(float));
  }

  return M;
}

void updateMatrixSlice(const arma::fmat& update, petuum::Table<float>& table,
                       int m, int n, int offset) {
  for (int j = 0; j < n; j++) {
    petuum::DenseUpdateBatch<float> batch(offset, m);

    std::memcpy(batch.get_mem(), update.colptr(j), m * sizeof(float));

    table.DenseBatchInc(j, batch);
  }
}
}
}
}
