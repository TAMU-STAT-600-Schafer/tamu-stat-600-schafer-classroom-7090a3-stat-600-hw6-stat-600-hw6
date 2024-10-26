// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include <RcppArmadillo.h>

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                            const arma::mat& M, int numIter = 100){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int n = X.n_rows;
    int p = X.n_cols;
    arma::uvec Y(n); // to store cluster assignments
    
    // Initialize any additional parameters if needed
    
    arma::mat X_1(n, p, arma::fill::zeros);
    X_1 += X;
    arma::mat M_1(K, p, arma::fill::zeros);
    M_1 += M;
    arma::mat M_2(K, p, arma::fill::zeros); 
    
    // For loop with kmeans algorithm
    for(int i = 0; i < numIter; i++){
      
      arma::colvec M_sqr = .5 * arma::sum(M % M, 1);
      arma::mat Norm(n, K);
      Norm = X_1 * M_1.t();
      
      for(int j = 0; j < K; j++){
        Norm.col(j) += M_sqr;
      }
      
      arma::colvec Y(n);
      
      for(int j = 0; j < n; j++){
        Y(j) = arma::index_max(Norm.row(j));
      }
      
      arma::colvec unique = arma::unique(Y);
      if(unique.n_elem < K){
        break;
      }
      
      
      
      
      
    }
    
    // Returns the vector of cluster assignments
    return(Y);
}

