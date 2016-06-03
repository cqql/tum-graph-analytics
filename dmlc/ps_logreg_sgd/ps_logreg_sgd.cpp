#include <algorithm>
#include <cmath>
#include <fstream>
#include <forward_list>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <armadillo>
#include <ps/ps.h>

using namespace ps;

typedef std::vector<double> stdvec;

class Worker {
    public:
        Worker() : worker(0) {};

        // Runs logistic regression using asynchronous stochastic gradient descent and l2 regularization
        // After training, a worker does testing
        void runLogReg(double rank, double totalWorkers, double regularCoeff, double learningRate, std::string trainPath, std::string testPath) {

			std::cout << "(Worker: " << rank << ")" << "starting" << std::endl;
            ps::Postoffice *office = ps::Postoffice::Get();
            std::cout << "(Worker: " << rank << ")" << "Office loaded" << std::endl;
            KVWorker<double> kv(rank);

            std::cout << "(Worker: " << rank << ")" << "Loading data" << std::endl;
            arma::mat data = ParseData(trainPath);

			
            int totalSamples = data.n_rows;
            int chunkSize = totalSamples * (1 / totalWorkers);
            int startIndex = totalSamples * (rank / totalWorkers);
            int endIndex = startIndex + chunkSize;
            if ((rank / totalWorkers) == 1)
            {
                endIndex = data.n_rows - 1;
            }

            int lastFeatureIndex = data.n_cols - 2;
            int labelIndex = data.n_cols - 1;

			std::cout << "(Worker: " << rank << ")" << startIndex << " " << labelIndex << " " << endIndex << "" << labelIndex << std::endl;
            //Split train data into features and labels
            //arma::mat labels = data.submat(startIndex, labelIndex, endIndex, labelIndex);
            arma::mat labels = data.submat(0, 0, 1, 1);
            
            //Normalise feature data
            //data = arma::normalise(data.submat(startIndex, 0, endIndex, lastFeatureIndex));
            data = arma::normalise(data.submat(0, 0, 1, 1));

            std::vector<ps::Key> keys(data.n_cols);
            std::iota (std::begin(keys), std::end(keys), 0);

            stdvec weights(data.n_cols, 0.0);

            //first worker initializes weights
            if (rank == 0)
            {
                int ts = kv.Push(keys, weights);
                kv.Wait(ts);
            }

            //Make sure that all workers start training at the same time
            office->Barrier(ps::kServerGroup | ps::kWorkerGroup);

            std::cout << "(Worker: " << rank << ")" << "Training starts" << std::endl;

            // Train
            for (int i = 0; i < data.n_rows; i++)
            {
                int ts = kv.Pull(keys, &weights);
                kv.Wait(ts);

                arma::mat armaWeights = arma::mat(weights);

                double classification = classify(armaWeights, data.row(i));
                double l2pen = l2Penalty(armaWeights);
                armaWeights = armaWeights + learningRate * ((labels(i, 0) - classification) - regularCoeff * l2pen);

                weights = arma::conv_to<stdvec>::from(armaWeights);

                ts = kv.Push(keys, weights);
                kv.Wait(ts);
            }

            //Make sure that all workers have finished training
            office->Barrier(ps::kServerGroup | ps::kWorkerGroup);

            //Get latest weights and print on screen
            int ts = kv.Pull(keys, &weights);
            kv.Wait(ts);

            std::cout << "(Worker: " << rank << ")" << "Testing starts" << std::endl;

            //Make sure that all workers start testing at the same time
            office->Barrier(ps::kServerGroup | ps::kWorkerGroup);

            data = ParseData(testPath);
            totalSamples = data.n_rows;
            chunkSize = totalSamples * (1 / totalWorkers);
            startIndex = totalSamples * (rank / totalWorkers);
            endIndex = startIndex + chunkSize;
            if ((rank / totalWorkers) == 1)
            {
                endIndex = data.n_rows - 1;
            }

            //Split test data into features and labels
            labels = data.submat(startIndex, labelIndex, endIndex, labelIndex);
            //Normalise feature data
            data = arma::normalise(data.submat(startIndex, 0, endIndex, lastFeatureIndex));

            ts = kv.Pull(keys, &weights);
            kv.Wait(ts);

            stdvec uniqueClasses = getClasses(labels);
            arma::mat armaWeights = arma::mat(weights);

            double correct = 0;
            double incorrect = 0;

            for (int i = 0; i < data.n_rows; i++)
            {
                double prediction = arma::dot(armaWeights, data.row(i));

                double predAccuracy = std::abs(labels(i, 0) - prediction) / labels(i, 0);

                if (predAccuracy >= 0.95)
                {
                    correct += 1;
                }
                else
                {
                    incorrect += 1;
                }
            }

            double modelAccuracy = correct / (correct + incorrect);
            std::vector<Key> accuKey(1);
            accuKey[0] = data.n_cols + 1 + rank;
            stdvec accuVal = { modelAccuracy };

            ts = kv.Push(accuKey, accuVal);
            kv.Wait(ts);


            //Make sure that all workers have finished testing
            office->Barrier(ps::kServerGroup | ps::kWorkerGroup);

            std::cout << "(Worker: " << rank << ")" << "Finished testing" << std::endl;

            std::vector<Key> accuKeys;
            stdvec accuVals;
            if (rank == 0)
            {
                for (int i = 0; i < totalWorkers; i++)
                {
                    accuKeys.push_back(data.n_cols + 1 + i);
                }

                ts = kv.Pull(accuKeys, &accuVals);
                kv.Wait(ts);

                double sum = std::accumulate(accuVals.begin(), accuVals.end(), 0);
                double count = accuVals.size();
                double finalAccuracy = sum / count;

                //Get latest weights and print ont screen
                ts = kv.Pull(keys, &weights);
                kv.Wait(ts);

                std::cout << "(Worker: " << rank << ")" << "Final accuracy: " << finalAccuracy << std::endl;
            }
        }

    private:
        ps::KVWorker<double> worker;

        arma::mat ParseData(std::string path) {
            arma::mat data;
            data.load(path);

            return data;
        };

        double classify(arma::mat weightsCol, arma::mat featuresRow) {
            double dotProduct = arma::dot(weightsCol, featuresRow);
            return sigmoid(dotProduct);
        }

        double l2Penalty(arma::mat weights) {
            return arma::dot(weights, weights);
        }

        double sigmoid(double input) {
            return 1.0 / (1.0 + std::exp(-input));
        }

        stdvec getClasses(arma::mat labels) {
            stdvec labelsVec = arma::conv_to<stdvec>::from(labels);
            sort( labelsVec.begin(), labelsVec.end());
            unique( labelsVec.begin(), labelsVec.end());

            return labelsVec;
        }
};

void StartServer() {
    auto server = new KVServer<double>(0);
    server->set_request_handle(KVServerDefaultHandle<double>());
    RegisterExitCallback([server]() {
        delete server;
    });
}

int main(int argc, char **argv) {

    int T = 100;
    double regularCoeff = 1.0d;
    double learningRate = 0.1d;
    std::string trainPath = "train.csv";
    std::string testPath = "test.csv";

    ps::Start();

    if (ps::IsScheduler()) {

    } else if (ps::IsServer()) {

        auto server = new KVServer<double>(0);
        server->set_request_handle(KVServerDefaultHandle<double>());
        ps::RegisterExitCallback([server]() {
            delete server;
        });

    } else if (ps::IsWorker()) {

        Worker worker;
        worker.runLogReg(ps::MyRank(), ps::NumWorkers(), regularCoeff, learningRate, trainPath, testPath);
    }

    ps::Finalize();

    return 0;
}
