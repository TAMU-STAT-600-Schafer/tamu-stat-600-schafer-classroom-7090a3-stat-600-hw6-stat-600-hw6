// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include <RcppArmadillo.h>
#include <stdlib.h>

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                      const arma::mat& M, int numIter = 100){
  // All input is assumed to be correct
  
  
  // Initialize some parameters
  int n = X.n_rows;
  int p = X.n_cols;
  arma::uvec Y(n); // to store cluster assignments
  
  // Initialize any additional parameters if needed
  
  arma::mat M_1 = M;
  arma::mat M_2(K, p, arma::fill::zeros); 
  arma::rowvec M_sum(p, arma::fill::zeros);
  arma::rowvec M_sqr;
  arma::mat Norm(n, K);
  arma::uvec unique;
  
  // For loop with kmeans algorithm
  for(int i = 0; i < numIter; i++){
    
    M_sqr = arma::sum(M_1 % M_1, 1).t() * .5;
    
    Norm = X * M_1.t();
    
    for(int j = 0; j < Norm.n_rows; j++){
      Norm.row(j) -= M_sqr;
    }
    
    for(int j = 0; j < n; j++){
      arma::rowvec norm_row = Norm.row(j);
      Y(j) = norm_row.index_max();
    }
    
    unique = arma::unique(Y);
    if(unique.n_elem < K){
      break;
    }
    
    for(int j =0; j < K; j++){
      arma::uvec Y_index = arma::find(Y == j); 
      arma::rowvec M_sum = arma::sum(X.rows(Y_index), 0);
      M_2.row(j)= M_sum * (1.0 / Y_index.n_elem);
    }
    
    if(arma::approx_equal(M_1, M_2, "absdiff", 1e-16)){
      return(Y);
    }
    
    M_1 = M_2;
  }
  
  // Returns the vector of cluster assignments
  return(Y);
}
