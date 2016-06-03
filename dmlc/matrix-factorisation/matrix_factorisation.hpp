

#ifndef matrix_factorisation_hpp
#define matrix_factorisation_hpp

#include <stdio.h>
#include <armadillo>
#include <stdlib.h>
#include <time.h>

#endif /* matrix_factorisation_hpp */

namespace factorisation{
	
	template<typename T>
	struct statistics{
		T g;
		arma::Col<T> bu;
		arma::Col<T> bv;
		
	};
	
	class SGD{
	public:
		void matrix_factorisation(double(*)(double, double),
								  int,
								  int,
								  double,
								  double);
		SGD(arma::sp_mat*, double);
		SGD(arma::sp_mat*);
		arma::mat U;
		arma::mat V;
		arma::sp_mat *data;
		statistics<double> st;
	};
	
}