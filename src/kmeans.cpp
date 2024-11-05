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
    
    // Calculate squared L2 norms of centroids (multiplied by 0.5 for optimization)
    M_sqr = arma::sum(M_1 % M_1, 1).t() * .5;
    
    // Dot products between data points and centroids for 
    // the squared Euclidean distance calculation
    Norm = X * M_1.t();
    
    // Complete the squared distance calculation by subtracting centroid norms
    // Uses the formula: ||x - m||^2 = ||x||^2 + ||m||^2 - 2x^T m
    for(int j = 0; j < Norm.n_rows; j++){
      Norm.row(j) -= M_sqr;
    }
    
    // Assign each point to its nearest centroid
    // index_max() is used because we negated distances in previous steps
    for(int j = 0; j < n; j++){
      Y_1(j) = Norm.row(j).index_max();
    }
    
    // Check if any clusters became empty. If empty, it breaks the loop
    unique = arma::unique(Y_1);
    if(unique.n_elem < K){
      break;
    }
    
    // Reset centroids to zero before recalculating
    M_1.zeros();
    
    // Sum up all points assigned to each cluster
    for(int j = 0; j < n; j++){
      M_1.row(Y_1(j)) += X.row(j);
    }
    
    // Calculate new centroids by averaging points in each cluster
    for(int j = 0; j < K; j++){
      arma::uvec Y_index = arma::find(Y_1 == j);  // find all points in cluster j
      M_1.row(j) /= Y_index.n_elem;               // divide by number of points to get mean
    }
    
    // Check for convergence
    if(arma::all(Y_1 == Y_2)) {
      return(Y_1);
    }
    
    // Store current assignments for next iteration's comparison
    Y_2 = Y_1;
  }
  
  // Return final cluster assignments after all iterations
  // This is reached if max iterations hit before convergence
  return(Y_1);
}