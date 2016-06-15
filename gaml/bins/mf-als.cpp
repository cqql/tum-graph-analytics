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

#include "../io/matrix_slice.h"
#include "../mf/gd/nn_projection.h"
#include "../mf/gd/worker.h"

enum RowType { FLOAT };

enum Table { P, U };

namespace po = boost::program_options;

struct MfalsThread {
  int id;
  std::string path;
  int k;
  int iterations;
  int minibatch;
  int seed;

  MfalsThread(int id, std::string path, int k, int iterations, int minibatch,
              int seed)
      : id(id),
        path(path),
        k(k),
        iterations(iterations),
        minibatch(minibatch),
        seed(seed) {}

  void run() {
    petuum::PSTableGroup::RegisterThread();

    std::ostringstream prodpath;
    std::ostringstream userpath;
    prodpath << this->path << "/rank-" << this->id << "-prod";
    userpath << this->path << "/rank-" << this->id << "-user";
    struct gaml::io::MatrixSlice prodms =
        gaml::io::MatrixSlice::parse(prodpath.str());
    struct gaml::io::MatrixSlice userms =
        gaml::io::MatrixSlice::parse(userpath.str());
    int pOffset = prodms.offset;
    int uOffset = userms.offset;
    auto pSlice = prodms.R;
    auto uSlice = userms.R;

    gaml::mf::gd::Worker worker(Table::P, Table::U, this->iterations,
                                this->minibatch, std::mt19937(this->seed),
                                gaml::mf::gd::NNProjection());
    auto factors = worker.factor(pSlice, pOffset, uSlice, uOffset, this->k);
    auto P = std::get<0>(factors);
    auto UT = std::get<1>(factors);

    if (this->id == 1) {
      P.save("out/P", arma::csv_ascii);
      UT.save("out/UT", arma::csv_ascii);
    }

    petuum::PSTableGroup::DeregisterThread();
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  po::options_description options;
  // clang-format off
  options.add_options()
      ("rank,r", po::value<int>()->default_value(7), "Rank of the factors")
      ("iterations,i", po::value<int>()->default_value(1),
       "Number of iterations")
      ("users", po::value<int>(), "Number of users")
      ("products", po::value<int>(), "Number of products")
      ("workers", po::value<int>()->default_value(3),
       "Number of workers")
      ("eval-rounds,e", po::value<int>()->default_value(0),
       "Eval the model every k rounds")
      ("minibatch,m", po::value<int>()->default_value(0),
       "Size of minibatch per worker")
      ("seed", po::value<int>()->default_value(0),
       "Random seed");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);

  int rank = vm["rank"].as<int>();
  int iterations = vm["iterations"].as<int>();
  int num_users = vm["users"].as<int>();
  int num_products = vm["products"].as<int>();
  int num_workers = vm["workers"].as<int>();
  int eval_rounds = vm["eval-rounds"].as<int>();
  int minibatch = vm["minibatch"].as<int>();
  int seed = vm["seed"].as<int>();

  // Register row types
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float>>(RowType::FLOAT);

  // Initialize group
  petuum::TableGroupConfig table_group_config;
  table_group_config.host_map.insert(
      std::make_pair(0, petuum::HostInfo(0, "127.0.0.1", "10000")));
  table_group_config.consistency_model = petuum::SSP;
  table_group_config.num_tables = 2;
  table_group_config.num_total_clients = 1;
  table_group_config.num_local_app_threads = num_workers + 1;
  // Somehow a larger number than 1 leads to hanging at the end while the main
  // thread waits for all server threads to terminate. Apparently one of them is
  // not receiving a kClientShutDown message.
  table_group_config.num_comm_channels_per_client = 1;
  table_group_config.client_id = 0;
  petuum::PSTableGroup::Init(table_group_config, false);

  // Create tables
  gaml::mf::gd::Worker::initTables(Table::P, Table::U, RowType::FLOAT, rank,
                                   num_products, num_users);

  petuum::PSTableGroup::CreateTableDone();

  std::vector<std::thread> threads(num_workers);

  // Run workers
  for (int i = 0; i < num_workers; i++) {
    threads[i] =
        std::thread(&MfalsThread::run,
                    std::unique_ptr<MfalsThread>(new MfalsThread(
                        i, "out", rank, iterations, minibatch, seed + i)));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Finalize
  petuum::PSTableGroup::ShutDown();

  return 0;
}
