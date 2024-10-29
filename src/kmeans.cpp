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
  arma::uvec Y_1(n); // to store cluster assignments
  arma::uvec Y_2(n); // to compare values between each iteration
  
  // Initialize any additional parameters if needed
  
  arma::mat M_1 = M; // make a copy of M matrix
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
      Y_1(j) = norm_row.index_max();
    }
    
    unique = arma::unique(Y_1);
    if(unique.n_elem < K){
      break;
    }
    
    M_1.zeros();
    for(int j = 0; j < n; j++){
      M_1.row(Y_1(j)) += X.row(j);
    }
    
    for(int j = 0; j < K; j++){
      arma::uvec Y_index = arma::find(Y_1 == j);
      M_1.row(j) /= Y_index.n_elem;
    }
    
    if(arma::all(Y_1 == Y_2)) {
      return(Y_1);
    }
    
    Y_2 = Y_1;
  }
  
  // Returns the vector of cluster assignments
  return(Y_1);
}