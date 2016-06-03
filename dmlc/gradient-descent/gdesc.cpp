#include <algorithm>
#include <cmath>
#include <fstream>
#include <forward_list>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <ps/ps.h>

// All of these have to be different from any ps::Command
enum struct GDescCommand {
  INITIALIZE = 10
};

template <typename T>
class ServerHandle {
public:
  void operator()(const ps::KVMeta &req_meta, const ps::KVPairs<T> &req_data,
                  ps::KVServer<T> *server) {
    size_t n = req_data.keys.size();
    ps::KVPairs<T> res;

    if (req_meta.cmd == GDescCommand::INITIALIZE) {

    } else if (req_meta.push) {
      if (!this->initialized) {
        this->grad.resize(n);
        std::fill(this->grad.begin(), this->grad.end(), 0.0);

        this->w.resize(n);
        std::fill(this->w.begin(), this->w.end(), 0.0);

        this->initialized = true;
      }

      for (int i = 0; i < n; i++) {
        this->grad[req_data.keys[i]] += req_data.vals[i];
      }

      this->numGradients++;

      if (this->numGradients == ps::NumWorkers()) {
        int N = this->w.size();

        double normw = 0.0;
        for (int i = 0; i < N; i++) {
          normw += this->w[i] * this->w[i];
        }
        normw = std::sqrt(normw);

        for (int i = 0; i < N; i++) {
          this->w[i] -= this->eta * this->grad[i];

          if (normw > 0) {
            this->w[i] -= this->lambda * this->w[i] / normw;
          }
        }

        // Reset gradients
        std::fill(this->grad.begin(), this->grad.end(), 0.0);
        this->numGradients = 0;
      }
    } else {
      res.keys = req_data.keys;
      res.vals.resize(n);

      for (int i = 0; i < n; i++) {
        res.vals[i] = this->w[req_data.keys[i]];
      }
    }

    server->Response(req_meta, res);
  }

private:
  int numGradients = 0;
  double eta = .15;
  double lambda = 0.001;
  bool initialized = false;
  std::vector<T> grad;
  std::vector<T> w;
};

class Server {
public:
  Server() : server(0) {
    this->server.set_request_handle(ServerHandle<double>());
  }

  void run(int T) {
    ps::Postoffice *office = ps::Postoffice::Get();

    for (int t = 0; t < T; t++) {
      office->Barrier(ps::kServerGroup | ps::kWorkerGroup);
    }

    office->Barrier(ps::kServerGroup | ps::kWorkerGroup);
  };

private:
  ps::KVServer<double> server;
};

struct Data {
  // Dimensionality of x
  size_t n;

  // Number of data points
  size_t N;

  // Input data
  std::vector<double> X;

  // Output data
  std::vector<double> y;
};

class Worker {
public:
  Worker() : worker(0){};

  void run(std::string path, int T) {
    const struct Data data = this->ParseData(path);
    std::vector<ps::Key> keys(data.n);
    std::vector<double> w(data.n, 0.0);
    std::vector<double> grad(data.n, 0.0);
    ps::Postoffice *office = ps::Postoffice::Get();

    std::iota(keys.begin(), keys.end(), 0);

    double mean = 0.0;
    for (int i = 0; i < data.N; i++) {
      mean += data.y[i];
    }
    mean /= data.N;

    // Push the dimensionality and mean
    ps::Message meanMsg;
    meanMsg.meta.simple_app = true;
    meanMsg.meta.request = true;
    meanMsg.meta.push = true;
    worker.Push({0, 1}, {(double)data.n, mean}, {}, GDescCommand::INITIALIZE);


    office->Barrier(ps::kServerGroup | ps::kWorkerGroup);

    for (int t = 0; t < T; t++) {
      // Pull the current weights
      int ts = this->worker.Pull(keys, &w);
      this->worker.Wait(ts);

      std::vector<double> error(data.N);
      for (int i = 0; i < data.N; i++) {
        error[i] = data.y[i] - std::inner_product(w.begin(), w.end(),
                                                  &data.X[i * data.n], 0.0);
      }

      // Gradient of the mean squared error
      for (int i = 0; i < data.n; i++) {
        grad[i] = 0.0;

        for (int j = 0; j < data.N; j++) {
          grad[i] -= data.X[j * data.n + i] * error[j];
        }

        grad[i] *= 1.0 / data.N;
      }

      double loss = 0.0;
      for (int i = 0; i < data.N; i++) {
        loss += error[i] * error[i];
      }
      loss /= data.N;
      std::cout << "Mean-Loss = " << loss << std::endl;

      // Normalize gradient
      double invnorm = 1 / std::sqrt(std::inner_product(
                               grad.begin(), grad.end(), grad.begin(), 0.0));

      if (invnorm < 1) {
        for (int i = 0; i < data.n; i++) {
          grad[i] *= invnorm;
        }
      }

      this->worker.Push(keys, grad);

      // Wait for all workers to push
      office->Barrier(ps::kServerGroup | ps::kWorkerGroup);
    }

    if (ps::MyRank() == 0) {
      int ts = this->worker.Pull(keys, &w);
      this->worker.Wait(ts);

      std::cout << "Weights:" << std::endl;
      for (int i = 0; i < data.n - 1; i++) {
        std::cout << w[i] << std::endl;
      }
      std::cout << std::endl << "Intercept: " << w[data.n - 1] << std::endl;
    }

    office->Barrier(ps::kServerGroup | ps::kWorkerGroup);
  };

private:
  ps::KVWorker<double> worker;

  struct Data ParseData(std::string path) {
    std::ifstream buffer(path);
    std::vector<double> x;
    std::vector<double> y;

    while (!buffer.eof()) {
      double tmp;
      buffer >> tmp;

      if (buffer.peek() == '\n') {
        x.push_back(1);
        y.push_back(tmp);
      } else {
        x.push_back(tmp);
      }
    }

    return {x.size() / y.size(), y.size(), x, y};
  };
};

int main(int argc, char **argv) {
  if (argc < 2) {
    return 1;
  }

  int T = 100;

  if (ps::IsScheduler()) {
    ps::Start();
  } else if (ps::IsServer()) {
    Server *server = new Server();

    ps::Start();

    server->run(T);

    // The server has to be destructed before finalize but the workers do not?!
    delete server;
  } else if (ps::IsWorker()) {
    ps::Start();

    std::ostringstream file;
    file << argv[1] << "-" << ps::MyRank();

    Worker worker;
    worker.run(file.str(), T);
  }

  ps::Finalize();

  return 0;
}
