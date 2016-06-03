
#include "matrix_factorisation.hpp"

namespace factorisation{
	
	SGD::SGD(arma::sp_mat *data, double g):data(data){
		st = statistics<double>();
		st.g = g;
		
		st.bu = arma::vec(data->n_rows, arma::fill::zeros);
		st.bv = arma::vec(data->n_cols, arma::fill::zeros);
		
	}
	
	SGD::SGD(arma::sp_mat *data):data(data){
		st = statistics<double>();
		st.bu = arma::vec(data->n_rows, arma::fill::zeros);
		st.bv = arma::vec(data->n_cols, arma::fill::zeros);
		
		auto r_nz = arma::Col<int>(data->n_rows, arma::fill::zeros);
		auto c_nz = arma::Col<int>(data->n_cols, arma::fill::zeros);
		
		auto end = data->end();
		for(auto it = data->begin(); it != end; ++it){
			auto row = it.row(); auto col = it.col();
			st.bu(row) += *it;
			st.bv(col) += *it;
			
			r_nz(row)++;
			c_nz(col)++;
		}

		r_nz.for_each([](int& val){
			val = val ? val : 1;
		});
		c_nz.for_each([](int& val){
			val = val ? val : 1;
		});
		
		st.g = st.bu.n_elem < st.bv.n_elem ? arma::accu(st.bu)/data->n_nonzero : arma::accu(st.bv)/data->n_nonzero;
		st.bu = st.bu / r_nz - st.g;
		st.bv = st.bv / c_nz - st.g;
	}
	
	void SGD::matrix_factorisation(double (*f)(double, double),
								   int n,
								   int rank,
								   double eta,
								   double lambda){
		
		srand (time(NULL));
		auto rows = data->n_rows;
		auto cols = data->n_cols;
		U = arma::randu<arma::mat>(rows, rank);
		V = arma::randu<arma::mat>(rank, cols);
		
		while(n--){
			//random edge (i,j)
			arma::sp_mat::const_iterator index = data->begin();
			int r = rand() % data->n_nonzero;
			std::advance(index, r);
			arma::uword i = index.row();
			arma::uword j = index.col();
			
			//prediction
			double pred = st.g + st.bu(i) + st.bv(j) + arma::dot(U.row(i), V.col(j));
			
			//compute error
			double err = f(pred, (*data)(i, j));
			
			//update biases
			st.g -= eta * (err + lambda * st.g);
			st.bu(i) -= eta * (err + lambda / data->row(i).n_nonzero * st.bu(i));
			st.bv(j) -= eta * (err + lambda / data->col(j).n_nonzero * st.bv(j));
			
			//update factors
			U.row(i) -= eta * (err * V.col(j).t() + lambda / data->row(i).n_nonzero * U.row(i));
			V.col(j) -= eta * (err * U.row(i).t() + lambda / data->col(j).n_nonzero * V.col(j));
		}
	}
}
