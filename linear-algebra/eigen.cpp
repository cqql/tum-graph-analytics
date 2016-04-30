#include <cmath>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "parse_dat.h"

int main(int argc, char** argv) {
  auto data = parse_dat("airfoil_self_noise.dat");
  auto x = data.first;
  auto y = data.second;
  int n = y.size();
  int m = x.size() / n;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X(n, m + 1);
  Eigen::VectorXd b(n);
  Eigen::VectorXd w = Eigen::VectorXd::Zero(m + 1);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      X(i, j) = x[i * m + j];
    }

    X(i, m) = 1.0;
    b(i) = y[i];
  }

  // Feature normalization
  Eigen::VectorXd ranges = X.colwise().maxCoeff() - X.colwise().minCoeff();
  //X = X.eval() * (1.0 / ranges);

  Eigen::VectorXd grad = Eigen::VectorXd::Zero(m + 1);

  int N = 0;
  while (true) {
    grad = -(X.transpose() * (b - X * w));

    if (std::abs(grad.norm()) < 1.0) {
      break;
    }

    w = w - grad;
    std::cout << w << std::endl;
  }

  std::cout << "Iterations: " << N << std::endl;
  std::cout << "Weights: " << std::endl << w << std::endl;
}
