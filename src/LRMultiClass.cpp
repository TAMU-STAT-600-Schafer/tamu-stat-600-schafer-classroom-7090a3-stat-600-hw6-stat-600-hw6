// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in right format)


// [[Rcpp::export]]
arma::mat softmax_matrix_c(const arma::mat& X, const arma::mat& beta) {
  // X: n * p
  // beta: p * K
  arma::mat Z = X * beta;  // n * K
  
  arma::vec Z_max(Z.n_rows);
  
  for(unsigned int i = 0; i < Z.n_rows; i++) {
    Z_max(i) = max(Z.row(i));
  }
  
  //std::cout << "softmax_matrix.each_row";

  arma::mat Z_exp = exp(Z.each_col() - Z_max);
  arma::vec Z_sum = sum(Z_exp, 1);  
  
  return Z_exp.each_col() / Z_sum;
}

// [[Rcpp::export]]
double loss_c(const arma::uvec& y, const arma::mat& P, const arma::mat& beta) {
  // beta p * K
  // P n * K
  double sum1 = 0;
  
  for(unsigned int k = 0; k < P.n_cols; k++) {
    for(unsigned int i = 0; i < y.n_elem; i++) {
      if(y(i) == k) {
        sum1 += log(P(i, k));
      }
    }
  }
  return -sum1 + accu(beta % beta);  // Note: sum(beta^2) in R becomes accu(beta % beta)
}


// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                          int numIter = 50, double eta = 0.1, double lambda = 1) {
  int p = X.n_cols;
  int K = arma::max(y) + 1;  
  
  arma::mat beta = beta_init;
  arma::vec objective(numIter + 1);
  
  
  arma::mat P = softmax_matrix_c(X, beta_init);
  
  
  objective(0) = loss_c(y, P, beta_init);
  
  
  for(int i = 0; i < numIter; i++) {
    for(int k = 0; k < K; k++) {
      
      arma::vec w = P.col(k) % (1 - P.col(k));
      
      arma::mat X_weighted = X.each_col() % w;
      //std::cout << "before X_weighted.each_row";
      
      //X_weighted.each_col() % w;
      //std::cout << "after X_weighted.each_row";
      
      arma::mat XtWX = X.t() * X_weighted;

      arma::vec y_indicator = arma::conv_to<arma::vec>::from(y == k);
      
      arma::vec grad = X.t() * (P.col(k) - y_indicator) + lambda * beta.col(k);
      
      arma::mat hessian = XtWX + lambda * arma::eye(p, p);
      
      beta.col(k) -= eta * solve(hessian, grad);
    }
    
    P = softmax_matrix_c(X, beta);
    
    objective(i + 1) = loss_c(y, P, beta);
  }
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("objective") = objective
  );
}