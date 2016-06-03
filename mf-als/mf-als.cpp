#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <armadillo>
#include <ps/ps.h>

struct MatrixData {
  int m;
  int n;
  int nnz;
  std::vector<unsigned int> rows;
  std::vector<unsigned int> cols;
  std::vector<float> vals;

  static struct MatrixData parse(std::string path) {
    struct MatrixData md;
    std::ifstream f(path, std::ios::binary);

    if (!f.is_open()) {
      throw std::invalid_argument("Could not open file");
    }

    f.read(reinterpret_cast<char*>(&md), 3 * sizeof(md.m));
    md.rows.resize(md.nnz);
    md.cols.resize(md.nnz);
    md.vals.resize(md.nnz);
    f.read(reinterpret_cast<char*>(md.rows.data()),
           md.nnz * sizeof(md.rows[0]));
    f.read(reinterpret_cast<char*>(md.cols.data()),
           md.nnz * sizeof(md.cols[0]));
    f.read(reinterpret_cast<char*>(md.vals.data()),
           md.nnz * sizeof(md.vals[0]));

    return md;
  }

  arma::sp_fmat toArma() {
    arma::urowvec rows(
        std::vector<arma::uword>(this->rows.begin(), this->rows.end()));
    arma::urowvec cols(
        std::vector<arma::uword>(this->cols.begin(), this->cols.end()));
    arma::umat locations = arma::join_cols(rows, cols);

    return arma::sp_fmat(locations, arma::fvec(this->vals));
  }
};

class MfWorker : public ps::SimpleApp {
 public:
  MfWorker() : ps::SimpleApp(0) {}

  Process(const ps::Message& msg) {}

 private:
  arma::sp_fmat Rprod;
  arma::sp_fmat Ruser;
  arma::mat P;
  arma::mat U;
};

int main(int argc, char** argv) {
  std::string basepath(argv[1]);
  int k = std::stoi(argv[2]);

  ps::Start();
  int rank = ps::MyRank();
  ps::Postoffice* office = ps::Postoffice::Get();

  if (ps::IsWorker()) {
    std::ostringstream prodspath;
    std::ostringstream userspath;
    prodspath << basepath << "/rank-" << rank << "-prods";
    userspath << basepath << "/rank-" << rank << "-users";
    struct MatrixData prodmd = MatrixData::parse(prodspath.str());
    auto Rprods = prodmd.toArma();
    struct MatrixData usermd = MatrixData::parse(userspath.str());
    auto Rusers = usermd.toArma();

    arma::mat P(usermd.m, k, arma::fill::randu);
    arma::mat U(k, prodmd.n, arma::fill::randu);

    MfWorker worker;

    if (rank == 0) {
    } else {
    }

    office->Barrier(ps::kWorkerGroup);
  }

  ps::Finalize();

  return 0;
}
