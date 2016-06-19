#include <fstream>
#include <iomanip>
#include <fenv.h>

#include <armadillo>
#include <glog/logging.h>

#include "worker.hpp"

#include "tensor_data.hpp"

#include "timer.hpp"

namespace tfals {

Worker::Worker(int id, std::string basepath, int rank, int iterations,
               int evalrounds, int usertableid, int prodtableid, int wordtableid)
    : id(id),
      basepath(basepath),
      rank(rank),
      iterations(iterations),
      evalrounds(evalrounds),
      usertableid(usertableid),
      prodtableid(prodtableid),
      wordtableid(wordtableid) {}

void Worker::run() {
  petuum::PSTableGroup::RegisterThread();

  std::ostringstream userpath;
  userpath << basepath << "/_user_train" << id;
  struct TensorData userdata = TensorData::parse(userpath.str());
  int useroffset = userdata.offset;
  auto Ruser = userdata.R;
  
  std::ostringstream prodpath;
  prodpath << basepath << "/_prod_train" << id;
  struct TensorData proddata = TensorData::parse(prodpath.str());
  int prodoffset = proddata.offset;
  auto Rprod = proddata.R;
  
  std::ostringstream wordpath;
  wordpath << basepath << "/_word_train" << id;
  struct TensorData worddata = TensorData::parse(wordpath.str());
  int wordoffset = worddata.offset;
  auto Rword = worddata.R;
  
  petuum::Table<float> usertable =
      petuum::PSTableGroup::GetTableOrDie<float>(usertableid);
  petuum::Table<float> prodtable =
      petuum::PSTableGroup::GetTableOrDie<float>(prodtableid);
  petuum::Table<float> wordtable =
      petuum::PSTableGroup::GetTableOrDie<float>(wordtableid);

  // Register rows
  if (this->id == 0) {
    for (int i = 0; i < this->rank; i++) {
      usertable.GetAsyncForced(i);
      prodtable.GetAsyncForced(i);
      wordtable.GetAsyncForced(i);
    }
  }

  petuum::PSTableGroup::GlobalBarrier();
  
  //LOG(INFO) << "Randomize U, P and T";
  arma::arma_rng::set_seed_random();
  if (id == 0) {
    randomizetable(usertable, rank, Rword.n_rows);
    randomizetable(prodtable, rank, Rword.n_cols);
    randomizetable(wordtable, rank, Ruser.n_words);
  }
  petuum::PSTableGroup::GlobalBarrier();
  
  //LOG(INFO) << "Fetch U, P and T on worker " << id;
  // Fetch U, P and T
  auto U = this->loadmat(usertable, Rword.n_rows, rank);
  auto P = this->loadmat(prodtable, Rword.n_cols, rank);
  auto T = this->loadmat(wordtable, Ruser.n_words, rank);

  arma::uvec Uz = arma::find(U == 0);
  arma::uvec Pz = arma::find(P == 0);
  arma::uvec Tz = arma::find(T == 0);
  
  if(Uz.n_elem != 0 || Pz.n_elem != 0 || Tz.n_elem != 0) {
    std::cout << Uz << std::endl;
    std::cout << Pz << std::endl;
    std::cout << Tz << std::endl;
  }
  //LOG(INFO) << "Start optimization";

  float step = 1.0;
  float lambda = 0.5;
  bool clamp = false; 
  bool reg = false;
  int reg_thr = 1;
  auto start = GetTimeMs64();
  auto end = GetTimeMs64();
  feenableexcept(FE_DIVBYZERO| FE_INVALID|FE_OVERFLOW);
  for (int i = 0; i < iterations; i++) {
     // Evaluate
    if (evalrounds > 0 && (i + 1) % evalrounds == 0) {
      if (id == 0) {
        std::cout << "Test => ";
        start = GetTimeMs64();
        evaltest(U, P, T);
      }

      if (id == 0) {
        std::cout << "Training => ";
        eval(U, P, T, Ruser, useroffset, 0);
        end = GetTimeMs64();
        std::cout << id << ": Eval " << end - start << std::endl;
      }
    }
    
    //LOG(INFO) << "Optimization round " << i << " on worker " << id;
    

    if (id == 0) {
      std::cout << "Round " << i + 1 << " with step length " << step
                << std::endl;
    }
/*
    ///////
    // Compute gradient for U, 1st variant
    ///////
    start = GetTimeMs64();
    arma::fmat Ugrad(Ruser.n_rows, rank, arma::fill::zeros);
    // iterate over all up pairs in Ruser
    for (std::size_t i = 0; i != Ruser.n_nz; ++i) {
      int userind = Ruser.rows[i];
      int prodind = Ruser.cols[i];
      auto wordbag = Ruser.getWordBagAt(i);
      
      arma::frowvec diff = (U.row(userind) % P.row(prodind)) * T.t() - wordbag;
      Ugrad.row(userind - useroffset) += P.row(prodind) % (diff * T);
    }
    Ugrad = arma::normalise(Ugrad, 2, 1);
    Ugrad = Ugrad * (-step);
    
    end = GetTimeMs64();
    std::cout << id << ": grad U " << end - start << std::endl;
*/    
    
    ///////
    // Compute gradient for U, 2nd variant
    ///////
    start = GetTimeMs64();
    arma::fmat Ugrad(Ruser.n_rows, rank, arma::fill::zeros);
    arma::fmat Ugrad1(Ruser.n_rows, rank, arma::fill::zeros);
    arma::fmat Ugrad2(Ruser.n_rows, rank, arma::fill::zeros);
    // iterate over all up pairs in Ruser
    for (std::size_t i = 0; i != Ruser.n_nz; ++i) {
      int userind = Ruser.rows[i];
      int prodind = Ruser.cols[i];
      auto wordbag = Ruser.getWordBagAt(i);
      
      Ugrad1.row(userind - useroffset) += P.row(prodind) % (wordbag * T);
      arma::fmat tmp = (T.t() * T);
      arma::frowvec tmp2 = P.row(prodind) % U.row(userind) % P.row(prodind);
      
      arma::uvec zer = arma::all(tmp, 1);
      bool zer2 = arma::all(tmp2);
      if(!arma::all(zer) || !zer2){
        arma::uvec tmp_z = arma::find(tmp==0);
        arma::uvec tmp2_z = arma::find(tmp2==0);
        std::cout << tmp_z << std::endl;
        std::cout << tmp2_z << std::endl;
      }
      Ugrad2.row(userind - useroffset) += tmp2 * tmp;
    }
    arma::fmat Ulocal = U.rows(useroffset, useroffset + Ruser.n_rows - 1);
    arma::uvec gz = arma::all(Ugrad2, 1);
    // having problems with 0 values
    if( !arma::all(gz)){
      std::cout << id << " Ugrad2 zero value" << std::endl;
      arma::uvec tmp = arma::find(Ugrad2==0);
      std::cout << tmp << std::endl;
      arma::clamp(Ugrad2, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    }
    Ugrad = (Ulocal % Ugrad1 / Ugrad2) - Ulocal;
    if(reg && i > reg_thr) {
      Ugrad = Ugrad - lambda * Ulocal % Ulocal / Ugrad2;
    }
    
    end = GetTimeMs64();
    std::cout << id << ": grad U " << end - start << std::endl;


    // Update U table
    updatetable(usertable, Ugrad, useroffset);

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated U
    U = loadmat(usertable, U.n_rows, U.n_cols);
    if(clamp){
      U = arma::clamp(U, 0.0, std::numeric_limits<float>::max());
    }
/*    
    ///////
    // Compute gradient for P
    ///////
    start = GetTimeMs64();
    arma::fmat Pgrad(Rprod.n_cols, rank, arma::fill::zeros);
    // iterate over all up pairs in Rprod
    for (std::size_t i = 0; i != Rprod.n_nz; ++i) {
      int userind = Rprod.rows[i];
      int prodind = Rprod.cols[i];
      auto wordbag = Rprod.getWordBagAt(i);
      
      arma::frowvec diff = (U.row(userind) % P.row(prodind)) * T.t() - wordbag;
      Pgrad.row(prodind - prodoffset) += U.row(userind) % (diff * T);
    }
    Pgrad = arma::normalise(Pgrad, 2, 1);
    Pgrad = Pgrad * (-step);
    
    end = GetTimeMs64();
    std::cout << id << ": grad P " << end - start << std::endl;
*/
    ///////
    // Compute gradient for P
    ///////
    start = GetTimeMs64();
    arma::fmat Pgrad(Rprod.n_cols, rank, arma::fill::zeros);
    arma::fmat Pgrad1(Rprod.n_cols, rank, arma::fill::zeros);
    arma::fmat Pgrad2(Rprod.n_cols, rank, arma::fill::zeros);
    // iterate over all up pairs in Rprod
    for (std::size_t i = 0; i != Rprod.n_nz; ++i) {
      int userind = Rprod.rows[i];
      int prodind = Rprod.cols[i];
      auto wordbag = Rprod.getWordBagAt(i);
      
      Pgrad1.row(prodind - prodoffset) += U.row(userind) % (wordbag * T);
      Pgrad2.row(prodind - prodoffset) += U.row(userind) % U.row(userind) % P.row(prodind) * (T.t() * T);
    }
    arma::uvec gz2 = arma::any(Pgrad2, 1);
    // having problems with 0 values
    if( arma::any(gz2==0)){
      std::cout << id << " Pgrad2 zero value" << std::endl;
      arma::uvec tmp = arma::find(Pgrad2==0);
      std::cout << tmp << std::endl;
      arma::clamp(Pgrad2, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    }
    
    arma::fmat Plocal = P.rows(prodoffset, prodoffset + Rprod.n_cols   - 1);
    Pgrad = (Plocal % Pgrad1 / Pgrad2) - Plocal;
    if(reg && i > reg_thr) {
      Pgrad = Pgrad - lambda * Plocal % Plocal / Pgrad2;
    }
    
    end = GetTimeMs64();
    std::cout << id << ": grad P " << end - start << std::endl;


    // Update P table
    updatetable(prodtable, Pgrad, prodoffset);

    petuum::PSTableGroup::GlobalBarrier();
  
    // Fetch updated P
    P = this->loadmat(prodtable, P.n_rows, P.n_cols);
    if(clamp) {
      P = arma::clamp(P, 0.0, std::numeric_limits<float>::max());
    }
/* 
    ///////
    // Compute gradient for T
    ///////
    start = GetTimeMs64();
    arma::fmat Tgrad(Rword.n_words, rank, arma::fill::zeros);
    // iterate over all uv pairs in Rword
    for (std::size_t i = 0; i != Rword.n_nz; ++i) {
      int userind = Rword.rows[i];
      int prodind = Rword.cols[i];
      
      auto wordbag = Rword.getWordBagAt(i);

      arma::frowvec diff = (U.row(userind) % P.row(prodind)) * T.rows(wordoffset, wordoffset + Rword.n_words - 1).t() - wordbag;

      for (int x = 0; x < rank; ++x) {
        Tgrad.col(x) += P(prodind, x) * U(userind, x) * diff.t();
      }
    }
    Tgrad = arma::normalise(Tgrad, 2, 1);
    Tgrad = Tgrad * (-step);
    
    end = GetTimeMs64();
    std::cout << id << ": grad T " << end - start << std::endl;
*/
    ///////
    // Compute gradient for T
    ///////
    start = GetTimeMs64();
    arma::fmat Tgrad(Rword.n_words, rank, arma::fill::zeros);
    arma::fmat Tgrad1(Rword.n_words, rank, arma::fill::zeros);
    arma::fmat Tgrad2(Rword.n_words, rank, arma::fill::zeros);
    arma::fmat Tlocal = T.rows(wordoffset, Rword.n_words + wordoffset - 1);
    // iterate over all uv pairs in Rword
    for (std::size_t i = 0; i != Rword.n_nz; ++i) {
      int userind = Rword.rows[i];
      int prodind = Rword.cols[i];
      
      auto wordbag = Rword.getWordBagAt(i);

      arma::frowvec tmp = (U.row(userind) % P.row(prodind)) * Tlocal.t();

      for (int x = 0; x < rank; ++x) {
        float p_tmp = P(prodind, x);
        float u_tmp = U(userind, x);
        Tgrad1.col(x) += P(prodind, x) * U(userind, x) * wordbag.t();
        Tgrad2.col(x) += P(prodind, x) * U(userind, x) * tmp.t();
      }
    }
    arma::uvec gz3 = arma::any(Tgrad2, 1);
    // having problems with 0 values
    if( arma::any(gz3==0)){
      std::cout << id << " Tgrad2 zero value" << std::endl;
      arma::uvec tmp = arma::find(Tgrad2==0);
      std::cout << tmp << std::endl;
      arma::clamp(Tgrad2, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    }
    
    Tgrad = (Tlocal % Tgrad1 / Tgrad2) - Tlocal;
    if(reg && i > reg_thr) {
      Tgrad = Tgrad - lambda * Tlocal % Tlocal / Tgrad2;
    }
    
    end = GetTimeMs64();
    std::cout << id << ": grad T " << end - start << std::endl;


    // Update T table
    updatetable(wordtable, Tgrad, wordoffset);

    petuum::PSTableGroup::GlobalBarrier();

    // Fetch updated T
    T = loadmat(wordtable, T.n_rows, T.n_cols);
    if(clamp) {
      T = arma::clamp(T, 0.0, std::numeric_limits<float>::max());
    }

    step *= 0.9;
  }

  // Evaluate (if not evaluated in last round)
  if (id == 0) {
    std::cout << "Test => ";
    evaltest(U, P, T);
    std::cout << "Training => ";
    eval(U, P, T, Ruser, useroffset, 0);

    U.save(basepath + "/U", arma::csv_ascii);
    P.save(basepath + "/P", arma::csv_ascii);
    T.save(basepath + "/T", arma::csv_ascii);
  }

  //LOG(INFO) << "Shutdown worker " << this->id;

  petuum::PSTableGroup::DeregisterThread();
}

// Initialize table as an m*n matrix with random entries
void Worker::randomizetable(petuum::Table<float>& table, int m, int n) {
  arma::fvec vec(n);
  for (int i = 0; i < m; i++) {
    vec.randn();
    vec = arma::abs(vec);

    petuum::DenseUpdateBatch<float> batch(0, n);
    std::memcpy(batch.get_mem(), vec.memptr(), n * sizeof(float));

    table.DenseBatchInc(i, batch);
  }
}

void Worker::updatetable(petuum::Table<float>& table, arma::fmat& grad, int offset) {
    for (int j = 0; j < grad.n_cols; j++) {
      petuum::DenseUpdateBatch<float> batch(offset, grad.n_rows);

      std::memcpy(batch.get_mem(), grad.colptr(j),
                  grad.n_rows * sizeof(float));

      table.DenseBatchInc(j, batch);
    }
}


arma::fmat Worker::loadmat(petuum::Table<float>& table, int m, int n) {
  arma::fmat M(m, n);
  petuum::RowAccessor rowacc;

  for (int i = 0; i < n; i++) {
    std::vector<float> tmp;
    const auto& col = table.Get<petuum::DenseRow<float>>(i, &rowacc);
    col.CopyToVector(&tmp);
    std::memcpy(M.colptr(i), tmp.data(), sizeof(float) * m);
  }
  return M;
}

void Worker::evaltest(arma::fmat& U, arma::fmat& P, arma::fmat& T) {
  std::ostringstream testpath;
  testpath << basepath << "/_test";
  auto testtd = TensorData::parse(testpath.str());
  auto Rtest = testtd.R;

  eval(U, P, T, Rtest, 0, 0);
}

void Worker::eval(arma::fmat& U, arma::fmat& P, arma::fmat& T, Sparse3dTensor& R,
                  int useroffset, int prodoffset) {
  float mse = 0;

  for (size_t i = 0; i != R.n_nz; ++i) {
    int userind = R.rows[i];
    int prodind = R.cols[i];

    arma::frowvec error =
        R.getWordBagAt(i) - (U.row(userind + useroffset) % P.row(prodind + prodoffset)) * T.t();
    mse += arma::norm(error);

    /*LOG(INFO) << "User " << std::setw(7) << userind + useroffset << ", Product "
              << std::setw(7) << prodind + prodoffset << ": " << std::setw(7)
              << error; //<< " (" << R.getWordBagAt(i) << ")";*/
  }

  std::cout << "MSE = " << mse / R.n_nz << std::endl;
}
}

