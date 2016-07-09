#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>

#include <armadillo>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <glog/logging.h>
#include <petuum_ps_common/include/petuum_ps.hpp>

#include "../io/matrix_slice.h"
#include "../mf/koren/worker.h"

enum RowType { FLOAT, INT };

enum Table { MU, BI, BU, ALPHA, Q, P, Y, SE };

namespace po = boost::program_options;

struct KorenThread {
  int nranks;
  int rank;
  int localRank;
  std::string path;
  int k;
  int iterations;
  int seed;
  float lambdab;
  float lambdaqpy;
  float gammab;
  float gammat;
  float gammaqpy;
  float beta;
  float atol;
  float rtol;
  float alpha;

  KorenThread(int nranks, int rank, int localRank, std::string path, int k,
              int iterations, int seed, float lambdab, float lambdaqpy,
              float gammab, float gammat, float gammaqpy, float beta,
              float atol, float rtol, float alpha)
      : nranks(nranks),
        rank(rank),
        localRank(localRank),
        path(path),
        k(k),
        iterations(iterations),
        seed(seed),
        lambdab(lambdab),
        lambdaqpy(lambdaqpy),
        gammab(gammab),
        gammat(gammat),
        gammaqpy(gammaqpy),
        beta(beta),
        atol(atol),
        rtol(rtol),
        alpha(alpha) {}

  void run() {
    petuum::PSTableGroup::RegisterThread();

    std::string itempath =
        this->path + "/rank-" + std::to_string(this->localRank) + "-prod";
    std::string userpath =
        this->path + "/rank-" + std::to_string(this->localRank) + "-user";
    std::string tUpath =
        this->path + "/rank-" + std::to_string(this->localRank) + "-user-time";
    struct gaml::io::MatrixSlice itemms =
        gaml::io::MatrixSlice::parse(itempath);
    struct gaml::io::MatrixSlice userms =
        gaml::io::MatrixSlice::parse(userpath);
    struct gaml::io::MatrixSlice timems = gaml::io::MatrixSlice::parse(tUpath);
    int iOffset = itemms.offset;
    int uOffset = userms.offset;
    auto iSlice = itemms.R;
    auto uSlice = userms.R;
    auto tSlice = timems.R;

    gaml::mf::koren::Worker worker(
        this->rank, this->nranks, this->lambdab, this->lambdaqpy, this->gammab,
        this->gammat, this->gammaqpy, this->beta, this->atol, this->rtol,
        Table::MU, Table::BI, Table::BU, Table::ALPHA, Table::Q, Table::P,
        Table::Y, Table::SE);
    auto factors = worker.factor(iSlice, iOffset, uSlice, tSlice, uOffset, k);
    auto mu = std::get<0>(factors);
    auto bi = std::get<1>(factors);
    auto bu = std::get<2>(factors);
    auto Q = std::get<3>(factors);
    auto P = std::get<4>(factors);
    auto Y = std::get<5>(factors);

    if (this->rank == 0) {
      arma::fvec({mu}).save(this->path + "/mu", arma::csv_ascii);
      bi.save(this->path + "/bi", arma::csv_ascii);
      bu.save(this->path + "/bu", arma::csv_ascii);
      Q.save(this->path + "/Q", arma::csv_ascii);
      P.save(this->path + "/P", arma::csv_ascii);
      Y.save(this->path + "/Y", arma::csv_ascii);
    }

    petuum::PSTableGroup::DeregisterThread();
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  po::options_description options;
  // clang-format off
  options.add_options()
      ("data", po::value<std::string>(), "Path to split data")
      ("clients", po::value<int>()->default_value(1), "Number of clients")
      ("id", po::value<int>()->default_value(0), "This client's ID")
      ("workers", po::value<int>()->default_value(3),
       "Number of worker threads per client")
      ("k,k", po::value<int>()->default_value(7), "Rank of the factors")
      ("iterations,i", po::value<int>()->default_value(1),
       "Number of iterations")
      ("users", po::value<int>(), "Number of users")
      ("items", po::value<int>(), "Number of items")
      ("seed", po::value<int>()->default_value(0), "Random seed")
      ("atol", po::value<float>()->default_value(0.01),
       "Minimum absolute MSE improvement before termination")
      ("rtol", po::value<float>()->default_value(0.01),
       "Minimum relative MSE improvement before termination")
      ("beta", po::value<float>()->default_value(0.4),
       "beta-parameters for time-dependent b_u")
      ("gamma", po::value<float>()->default_value(1.0),
       "GD step length")
      ("gamma-b", po::value<float>()->default_value(0.007),
       "GD step length for the biases")
      ("gamma-t", po::value<float>()->default_value(0.007),
       "GD step length for temporal dynamics")
      ("gamma-qpy", po::value<float>()->default_value(0.007),
       "GD step length for the factors")
      ("lambda", po::value<float>()->default_value(1.0),
       "Regularization weight")
      ("lambda-b", po::value<float>()->default_value(0.005),
       "Regularization weight for the biases")
      ("lambda-qpy", po::value<float>()->default_value(0.015),
       "Regularization weight for the factors");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);

  std::string datapath = vm["data"].as<std::string>();
  int client_id = vm["id"].as<int>();
  int num_clients = vm["clients"].as<int>();
  int num_workers = vm["workers"].as<int>();
  int k = vm["k"].as<int>();
  int iterations = vm["iterations"].as<int>();
  int nUsers = vm["users"].as<int>();
  int nItems = vm["items"].as<int>();
  int seed = vm["seed"].as<int>();
  float atol = vm["atol"].as<float>();
  float rtol = vm["rtol"].as<float>();
  float beta = vm["beta"].as<float>();
  float gamma = vm["gamma"].as<float>();
  float gammab = vm["gamma-b"].as<float>();
  float gammat = vm["gamma-t"].as<float>();
  float gammaqpy = vm["gamma-qpy"].as<float>();
  float lambda = vm["lambda"].as<float>();
  float lambdab = vm["lambda-b"].as<float>();
  float lambdaqpy = vm["lambda-qpy"].as<float>();

  int nranks = num_clients * num_workers;

  // Register row types
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float>>(RowType::FLOAT);
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<int>>(RowType::INT);

  // Initialize group
  petuum::TableGroupConfig table_group_config;
  table_group_config.host_map.insert(
      std::make_pair(0, petuum::HostInfo(0, "127.0.0.1", "10000")));
  table_group_config.consistency_model = petuum::SSP;
  table_group_config.num_tables = 8;
  table_group_config.num_total_clients = num_clients;
  table_group_config.num_local_app_threads = num_workers + 1;
  // Somehow a larger number than 1 leads to hanging at the end while the main
  // thread waits for all server threads to terminate. Apparently one of them is
  // not receiving a kClientShutDown message.
  table_group_config.num_comm_channels_per_client = 1;
  table_group_config.client_id = client_id;
  petuum::PSTableGroup::Init(table_group_config, false);

  // Create tables
  gaml::mf::koren::Worker::initTables(Table::MU, Table::BI, Table::BU,
                                      Table::ALPHA, Table::Q, Table::P,
                                      Table::Y, Table::SE, RowType::FLOAT,
                                      RowType::INT, k, nItems, nUsers, nranks);

  petuum::PSTableGroup::CreateTableDone();

  std::vector<std::thread> threads(num_workers);

  // Run workers
  for (int i = 0; i < num_workers; i++) {
    int rank = client_id * num_workers + i;

    threads[i] = std::thread(
        &KorenThread::run,
        std::unique_ptr<KorenThread>(new KorenThread(
            nranks, rank, i, datapath, k, iterations, seed + i, lambdab,
            lambdaqpy, gammab, gammat, gammaqpy, beta, atol, rtol, 0.0)));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Finalize
  petuum::PSTableGroup::ShutDown();

  return 0;
}
