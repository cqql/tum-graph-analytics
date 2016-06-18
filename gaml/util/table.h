#ifndef GAML_UTIL_TABLE_H_
#define GAML_UTIL_TABLE_H_

#include <armadillo>
#include <petuum_ps_common/include/petuum_ps.hpp>

namespace gaml {

namespace util {

namespace table {

/**
 * Load the complete n*m table as an m*n matrix
 */
arma::fmat loadMatrix(petuum::Table<float>& table, int m, int n);

/**
 * Apply the differences in `update` to `table`
 *
 * `update` is an m*n matrix and it is applied to `table` with an offset from
 * the left of `offset`.
 */
void updateMatrixSlice(const arma::fmat& update, petuum::Table<float>& table,
                       int m, int n, int offset);
}
}
}

#endif  // GAML_UTIL_TABLE_H_
