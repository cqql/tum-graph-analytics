#include <cmath>
#include <iostream>

#include <armadillo>

#include "parse_dat.h"

int main(int argc, char **argv) {
  auto data = parse_dat("airfoil_self_noise.dat");
  auto datax = data.first;
  auto datay = data.second;
  int n = datay.size();
  int m = datax.size() / n;

  // The data is read as a row vector so we have to do some extra work when
  // initializing X because matrices are initialized columnwise
  arma::rowvec rowX(const_cast<const double*>(datax.data()), datax.size());
  arma::mat X(rowX.t());
  X.reshape(m, n);
  X = X.t();
  arma::vec y(const_cast<const double*>(datay.data()), n);

  // Normalize features
  arma::rowvec ranges = arma::max(X, 0) - arma::min(X, 0);
  X.each_row() /= ranges;

  // Append column of ones
  X = arma::join_horiz(X, arma::vec(n, arma::fill::ones));

  arma::vec w(m + 1, arma::fill::zeros);
  arma::vec grad(m + 1);

  int N = 0;
  while (true) {
    grad = X.t() * (X * w - y);

    double t = 2.0;
    double currloss = arma::sum(arma::pow(y - X * w, 2));
    double newloss = 0.0;

    // Line search
    double ngrad = arma::norm(grad);
    do {
      t /= 2;
      newloss = arma::sum(arma::pow(y - X * (w - t * grad), 2));
    } while (newloss > currloss - t * ngrad);

    w -= t * grad;

    if (currloss - newloss < 0.0000000001) {
      break;
    }

    N++;
  }

  // Scale weights
  w /= (arma::join_rows(ranges, arma::rowvec{1.0})).t();

  std::cout << "Iterations: " << N << std::endl;
  std::cout << "Loss: " << arma::sum(arma::pow(y - X * w, 2)) << std::endl;
  std::cout << "Weights: " << std::endl << w;

  return 0;
}
