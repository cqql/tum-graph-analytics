#ifndef GAML_MF_WORKER_H_
#define GAML_MF_WORKER_H_

#include <armadillo>

namespace gaml {

namespace mf {

/**
 * Bosen worker for matrix factorization
 *
 *
 */
class Worker {
 public:
  // The factoring results
  arma::fmat P;
  arma::fmat UT;

  Worker(int pTableId, int uTableId, int iterations, int k, int pOffset,
         arma::sp_fmat pSlice, int uOffset, arma::sp_fmat uSlice)
      : pTableId(pTableId),
        uTableId(uTableId),
        iterations(iterations),
        k(k),
        pOffset(pOffset),
        pSlice(pSlice),
        uOffset(uOffset),
        uSlice(uSlice) {}

  void run();

  static void initTables(int pTableId, int uTableId, int rowType, int k,
                         int pNumRows, int uNumRows);

 private:
  const int pTableId;
  const int uTableId;
  const int iterations;
  const int k;

  // Slice of R along the P side
  const int pOffset;
  const arma::sp_fmat pSlice;

  // Slice of R along the U side
  const int uOffset;
  const arma::sp_fmat uSlice;

  /**
   * Initialize the m*n submatrix with offset from the left of table randomly
   */
  void randomizeTable(petuum::Table<float> table, int m, int n, int offset);

  /**
   * Load the complete n*m table as an m*n matrix
   */
  arma::fmat loadMatrix(petuum::Table<float>& table, int m, int n);
};
}
}

#endif  // GAML_MF_WORKER_H_
