#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>
#include <utility>

#include <armadillo>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <glog/logging.h>
#include <petuum_ps_common/include/petuum_ps.hpp>

#include "matrix_data.h"
#include "worker.h"

enum RowType { FLOAT };

enum Table { P, U };

namespace po = boost::program_options;

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
       "Number of workers");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, options), vm);

  int rank = vm["rank"].as<int>();
  int iterations = vm["iterations"].as<int>();
  int num_users = vm["users"].as<int>();
  int num_products = vm["products"].as<int>();
  int num_workers = vm["workers"].as<int>();

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
  petuum::ClientTableConfig p_config;
  p_config.table_info.row_type = RowType::FLOAT;
  p_config.table_info.row_capacity = num_products;
  p_config.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  p_config.table_info.table_staleness = 0;
  p_config.table_info.oplog_dense_serialized = true;
  p_config.table_info.dense_row_oplog_capacity =
      p_config.table_info.row_capacity;
  p_config.process_cache_capacity = rank;
  p_config.oplog_capacity = rank;
  p_config.thread_cache_capacity = 1;
  p_config.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable((int)Table::P, p_config);

  petuum::ClientTableConfig u_config;
  u_config.table_info.row_type = RowType::FLOAT;
  u_config.table_info.row_capacity = num_users;
  u_config.table_info.row_oplog_type = petuum::RowOpLogType::kDenseRowOpLog;
  u_config.table_info.table_staleness = 0;
  u_config.table_info.oplog_dense_serialized = true;
  u_config.table_info.dense_row_oplog_capacity =
      u_config.table_info.row_capacity;
  u_config.process_cache_capacity = rank;
  u_config.oplog_capacity = rank;
  u_config.thread_cache_capacity = 1;
  u_config.process_storage_type = petuum::BoundedDense;
  petuum::PSTableGroup::CreateTable((int)Table::U, u_config);

  petuum::PSTableGroup::CreateTableDone();

  std::vector<std::thread> threads(num_workers);

  // Run workers
  for (int i = 0; i < num_workers; i++) {
    threads[i] =
        std::thread(&mfals::Worker::run,
                    std::unique_ptr<mfals::Worker>(new mfals::Worker(
                        i, "out", rank, iterations, Table::P, Table::U)));
  }

  for (auto& thread : threads) {
    thread.join();
    LOG(INFO) << "Join";
  }

  // Finalize
  petuum::PSTableGroup::ShutDown();

  LOG(INFO) << "Shutdown";

  return 0;
}
