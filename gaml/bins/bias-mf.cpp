#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <utility>

#include <armadillo>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <glog/logging.h>
#include <petuum_ps_common/include/petuum_ps.hpp>

#include "../biases/worker.h"
#include "../io/matrix_slice.h"
#include "../mf/als/worker.h"
#include "../mf/gd/nn_projection.h"
#include "../mf/gd/worker.h"

enum RowType { FLOAT };

enum Table { P, U, SE, P2, U2, MEANS };

namespace po = boost::program_options;

struct BiasMfThread {
  int nranks;
  int rank;
  int localRank;
  std::string path;
  int k;
  int iterations;
  int minibatch;
  int seed;
  float atol;
  float rtol;
  float alpha;

  BiasMfThread(int nranks, int rank, int localRank, std::string path, int k,
               int iterations, int minibatch, int seed, float atol, float rtol,
               float alpha)
      : nranks(nranks),
        rank(rank),
        localRank(localRank),
        path(path),
        k(k),
        iterations(iterations),
        minibatch(minibatch),
        seed(seed),
        atol(atol),
        rtol(rtol),
        alpha(alpha) {}

  void run() {
    petuum::PSTableGroup::RegisterThread();

    std::ostringstream prodpath;
    std::ostringstream userpath;
    prodpath << this->path << "/rank-" << this->localRank << "-prod";
    userpath << this->path << "/rank-" << this->localRank << "-user";
    struct gaml::io::MatrixSlice prodms =
        gaml::io::MatrixSlice::parse(prodpath.str());
    struct gaml::io::MatrixSlice userms =
        gaml::io::MatrixSlice::parse(userpath.str());
    int pOffset = prodms.offset;
    int uOffset = userms.offset;
    auto pSlice = prodms.R;
    auto uSlice = userms.R;

    gaml::biases::Worker biasworker(Table::P2, Table::U2, Table::MEANS,
                                    this->nranks, this->rank);
    gaml::mf::als::Worker mfworker(Table::P, Table::U, Table::SE, this->nranks,
                                   this->rank, this->atol, this->rtol,
                                   this->alpha);
    auto biases = biasworker.compute(pSlice, pOffset, uSlice, uOffset);
    auto mean = std::get<0>(biases);
    auto umeans = std::get<1>(biases);
    auto pmeans = std::get<2>(biases);
    auto uunbiased = std::get<3>(biases);
    auto punbiased = std::get<4>(biases);
    auto factors =
        mfworker.factor(punbiased, pOffset, uunbiased, uOffset, this->k);
    auto P = std::get<0>(factors);
    auto UT = std::get<1>(factors);

    if (this->rank == 0) {
      arma::fvec({mean}).save(this->path + "/mu", arma::csv_ascii);
      umeans.save(this->path + "/bu", arma::csv_ascii);
      pmeans.save(this->path + "/bi", arma::csv_ascii);
      P.t().eval().save(this->path + "/Q", arma::csv_ascii);
      UT.t().eval().save(this->path + "/P", arma::csv_ascii);
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
      ("products", po::value<int>(), "Number of products")
      ("eval-rounds,e", po::value<int>()->default_value(0),
       "Eval the model every k rounds")
      ("minibatch,m", po::value<int>()->default_value(0),
       "Size of minibatch per worker")
      ("seed", po::value<int>()->default_value(0),
       "Random seed")
      ("atol", po::value<float>()->default_value(0.01),
       "Minimum absolute MSE improvement before termination")
      ("rtol", po::value<float>()->default_value(0.01),
       "Minimum relative MSE improvement before termination")
      ("alpha", po::value<float>()->default_value(1.0),
       "Ridge regression weight");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);

  std::string datapath = vm["data"].as<std::string>();
  int client_id = vm["id"].as<int>();
  int num_clients = vm["clients"].as<int>();
  int num_workers = vm["workers"].as<int>();
  int k = vm["k"].as<int>();
  int iterations = vm["iterations"].as<int>();
  int num_users = vm["users"].as<int>();
  int num_products = vm["products"].as<int>();
  int eval_rounds = vm["eval-rounds"].as<int>();
  int minibatch = vm["minibatch"].as<int>();
  int seed = vm["seed"].as<int>();
  float atol = vm["atol"].as<float>();
  float rtol = vm["rtol"].as<float>();
  float alpha = vm["alpha"].as<float>();

  int nranks = num_clients * num_workers;

  // Register row types
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float>>(RowType::FLOAT);

  // Initialize group
  petuum::TableGroupConfig table_group_config;
  table_group_config.host_map.insert(
      std::make_pair(0, petuum::HostInfo(0, "127.0.0.1", "10000")));
  table_group_config.consistency_model = petuum::SSP;
  table_group_config.num_tables = 6;
  table_group_config.num_total_clients = num_clients;
  table_group_config.num_local_app_threads = num_workers + 1;
  // Somehow a larger number than 1 leads to hanging at the end while the main
  // thread waits for all server threads to terminate. Apparently one of them is
  // not receiving a kClientShutDown message.
  table_group_config.num_comm_channels_per_client = 1;
  table_group_config.client_id = client_id;
  petuum::PSTableGroup::Init(table_group_config, false);

  // Create tables
  gaml::mf::als::Worker::initTables(Table::P, Table::U, RowType::FLOAT, k,
                                    num_products, num_users, Table::SE, nranks);
  gaml::biases::Worker::initTables(RowType::FLOAT, Table::P2, num_products,
                                   Table::U2, num_users, Table::MEANS, nranks);

  petuum::PSTableGroup::CreateTableDone();

  std::vector<std::thread> threads(num_workers);

  // Run workers
  for (int i = 0; i < num_workers; i++) {
    int rank = client_id * num_workers + i;

    threads[i] = std::thread(&BiasMfThread::run,
                             std::unique_ptr<BiasMfThread>(new BiasMfThread(
                                 nranks, rank, i, datapath, k, iterations,
                                 minibatch, seed + i, atol, rtol, alpha)));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Finalize
  petuum::PSTableGroup::ShutDown();

  return 0;
}
