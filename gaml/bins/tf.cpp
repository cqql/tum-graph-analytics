#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>
#include <string>

#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#include <armadillo>
#include <glog/logging.h>
#include <boost/program_options.hpp>
#include <petuum_ps_common/include/petuum_ps.hpp>

#include "../io/tensor.hpp"
#include "../tf/worker.hpp"

enum RowType { FLOAT };

enum Table { U, P, T, SE};

namespace po = boost::program_options;

struct TfThread {
  int id;
  std::string path;
  int rank;
  int iterations;
  int evalrounds;
  int seed;
  float atol;
  float rtol;
  float alpha;
  float lambda;
  bool clamp;
  bool reg;
  int reg_thr;

  TfThread(int id, std::string path, int rank, int iterations,
               int evalrounds, float lambda, bool clamp, bool reg, int reg_thr)
    : id(id),
      path(path),
      rank(rank),
      iterations(iterations),
      evalrounds(evalrounds),
      lambda(lambda),
      clamp(clamp),
      reg(reg),
      reg_thr(reg_thr) {}

  void run() {
    petuum::PSTableGroup::RegisterThread();
    
    std::ostringstream userpath;
    std::ostringstream prodpath;
    std::ostringstream wordpath;
    
    userpath << path << "/_user_train" << id;
    prodpath << path << "/_prod_train" << id;
    wordpath << path << "/_word_train" << id;
    
    auto userdata = gaml::io::TensorSlice::parse(userpath.str());
    auto proddata = gaml::io::TensorSlice::parse(prodpath.str());
    auto worddata = gaml::io::TensorSlice::parse(wordpath.str());
    
    int useroffset = userdata.offset;
    int prodoffset = proddata.offset;
    int wordoffset = worddata.offset;
    
    const auto Ruser = userdata.R;
    const auto Rprod = proddata.R;
    const auto Rword = worddata.R;
    std::tuple<arma::fmat, arma::fmat, arma::fmat> factors;
    
    if(id == 0) {
      std::ostringstream testpath;
      testpath << path << "/_test";
      auto testTensor = gaml::io::TensorSlice::parse(testpath.str());
      const auto Rtest = testTensor.R;
      gaml::tf::Worker worker(id, rank, iterations, Table::U, Table::P, Table::T, Table::SE, 
                            useroffset, prodoffset, wordoffset, 
                            Ruser, Rprod, Rword, Rtest);
      
      factors = worker.factorize(lambda, clamp, reg, reg_thr);
    } else {
      gaml::tf::Worker worker(id, rank, iterations, Table::U, Table::P, Table::T, Table::SE, 
                            useroffset, prodoffset, wordoffset, 
                            Ruser, Rprod, Rword);
      factors = worker.factorize(lambda, clamp, reg, reg_thr);
    }
    auto U = std::get<0>(factors);
    auto P = std::get<1>(factors);
    auto T = std::get<2>(factors);
    
    if(id == 0) {
      std::ostringstream Upath;
      std::ostringstream Ppath;
      std::ostringstream Tpath;
      Upath << this->path << "/U";
      Ppath << this->path << "/P";
      Tpath << this->path << "/T";
      U.save(Upath.str(), arma::csv_ascii);
      P.save(Ppath.str(), arma::csv_ascii);
      T.save(Tpath.str(), arma::csv_ascii);
    }
    
    petuum::PSTableGroup::DeregisterThread();
  }
};


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // Declare a group of options that will be allowed
  // only on the command line
  po::options_description generic("Generic options");
  generic.add_options()
      ("data", po::value<std::string>(), "Path to split data")
      ("rank,k", po::value<int>()->default_value(7), "Rank of the factors")
      ;
      
  // Declare a group of options that will be allowed
  // both on the command line and in a config file
  po::options_description config("Configuration");
  config.add_options()
      ("clients", po::value<int>()->default_value(1), "Number of clients")
      ("id", po::value<int>()->default_value(0), "This client's ID")
      ("workers", po::value<int>()->default_value(3),
       "Number of worker threads per client")
      ("iterations,i", po::value<int>()->default_value(1),
       "Number of iterations")
      ("users", po::value<int>(), "Number of users")
      ("products", po::value<int>(), "Number of products")
      ("words", po::value<int>()->default_value(2938), "Number of words")
      ("eval-rounds,e", po::value<int>()->default_value(0),
       "Eval the model every k rounds")
      ("seed", po::value<int>()->default_value(0),
       "Random seed")
      ("atol", po::value<float>()->default_value(0.01),
       "Minimum absolute MSE improvement before termination")
      ("rtol", po::value<float>()->default_value(0.01),
       "Minimum relative MSE improvement before termination")
      ("clamp", po::value<bool>()->default_value(false),
       "If true, then use non-negative projection.")
      ("reg", po::value<bool>()->default_value(false),
       "If true, then regularize.")
      ("reg-thr", po::value<int>()->default_value(1),
       "Iteration in which to start regularize")
      ("lambda", po::value<float>()->default_value(0.5),
       "Weight of the l2 regularizer")
      ;
       
  po::options_description cmdline_options;
  cmdline_options.add(generic).add(config);
  
  po::options_description config_file_options;
  config_file_options.add(config);

  po::variables_map vm;
  store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
  
  std::string datapath = vm["data"].as<std::string>();
  int rank = vm["rank"].as<int>();
  
  std::ifstream ifs(datapath + "/run_config.cfg");
  store(parse_config_file(ifs, config_file_options), vm);
  //notify(vm);
  
  int client_id = vm["id"].as<int>();
  int num_clients = vm["clients"].as<int>();
  int num_workers = vm["workers"].as<int>();
  int iterations = vm["iterations"].as<int>();
  int num_users = vm["users"].as<int>();
  int num_products = vm["products"].as<int>();
  int num_words = vm["words"].as<int>();
  int eval_rounds = vm["eval-rounds"].as<int>();
  int seed = vm["seed"].as<int>();
  float atol = vm["atol"].as<float>();
  float rtol = vm["rtol"].as<float>();
  bool clamp = vm["clamp"].as<bool>();
  bool reg = vm["reg"].as<bool>();
  int reg_thr = vm["reg-thr"].as<int>();
  float lambda = vm["lambda"].as<float>();

  // Register row types
  petuum::PSTableGroup::RegisterRow<petuum::DenseRow<float>>(RowType::FLOAT);

  // Initialize group
  petuum::TableGroupConfig table_group_config;
  table_group_config.host_map.insert(
      std::make_pair(0, petuum::HostInfo(0, "127.0.0.1", "10000")));
  table_group_config.consistency_model = petuum::SSP;
  table_group_config.num_tables = 4;
  table_group_config.num_total_clients = num_clients;
  table_group_config.num_local_app_threads = num_workers + 1;
  // Somehow a larger number than 1 leads to hanging at the end while the main
  // thread waits for all seRproder threads to terminate. Apparently one of them is
  // not receiving a kClientShutDown message.
  table_group_config.num_comm_channels_per_client = 1;
  table_group_config.client_id = client_id;
  petuum::PSTableGroup::Init(table_group_config, false);

  // create tables
  gaml::tf::Worker::initTables(Table::U, Table::P, Table::T, Table::SE, RowType::FLOAT, rank,
                               num_users, num_products, num_words, iterations, num_workers);
  petuum::PSTableGroup::CreateTableDone();

  std::vector<std::thread> threads(num_workers);

  // run workers
  for (int i = 0; i < num_workers; i++) {
    threads[i] = std::thread(&TfThread::run,
                             std::unique_ptr<TfThread>(new TfThread(
                              i, datapath, rank, iterations, eval_rounds, lambda, clamp, reg, reg_thr)));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // Finalize
  petuum::PSTableGroup::ShutDown();

  return 0;
}

