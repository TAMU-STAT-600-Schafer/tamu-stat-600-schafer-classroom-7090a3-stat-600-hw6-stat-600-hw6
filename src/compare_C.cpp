// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
#include <RcppArmadillo.h>
#include <math.h>

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
  
  // Initialize Z_max vector
  arma::vec Z_max(Z.n_rows);
  
  // Get maximum for each row using a loop (to match R implementation)
  for(unsigned int i = 0; i < Z.n_rows; i++) {
    Z_max(i) = max(Z.row(i));
  }
  
  // Calculate exponentials with numerical stability
  std::cout << "softmax_matrix.each_row";
  
  arma::mat Z_exp = exp(Z.each_col() - Z_max);
  arma::vec Z_sum = sum(Z_exp, 1);  // rowSums equivalent
  
  // Return probabilities
  return Z_exp.each_col() / Z_sum;
}

// [[Rcpp::export]]
double loss_c(const arma::uvec& y, const arma::mat& P, const arma::mat& beta, const double lambda) {
  // beta p * K
  // P n * K
  double sum1 = 0;
  
  // Sum of log likelihood
  for(unsigned int k = 0; k < P.n_cols; k++) {
    for(unsigned int i = 0; i < y.n_elem; i++) {
      if(y(i) == k) {
        sum1 += std::log(P(i, k));
      }
    }
  }
  return -sum1 + arma::dot(beta, beta) * (lambda / 2);  // Note: sum(beta^2) in R becomes accu(beta % beta)
}


// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                          int numIter = 50, double eta = 0.1, double lambda = 1) {
  int n = X.n_rows;
  int p = X.n_cols;
  int K = arma::max(y) + 1;  // number of classes
  
  arma::mat beta = beta_init;
  arma::vec objective(numIter + 1);
  
  // Calculate initial probabilities
  arma::mat P = softmax_matrix_c(X, beta_init);
  
  // Calculate initial objective value
  objective(0) = loss_c(y, P, beta_init, lambda);
  
  // Newton's method cycle
  for(int i = 0; i < numIter; i++) {
    for(int k = 0; k < K; k++) {
      // Calculate weights
      arma::vec w = P.col(k) % (1 - P.col(k));
      
      // Create weighted X (matching R's sweep operation)
      arma::mat X_weighted = X;
      std::cout << "before X_weighted.each_row";
      
      for(int j = 0; j < n; j++){
        X_weighted.row(j) *= w(j);
      }
      
      // Calculate XtWX
      arma::mat XtWX = X.t() * X_weighted;
      
      // Create indicator vector for current class (k-1 to match R indexing)
      arma::vec y_indicator = arma::conv_to<arma::vec>::from(y == k);
      
      // Calculate gradient and add regularization
      arma::vec grad = X.t() * (P.col(k) - y_indicator) + lambda * beta.col(k);
      
      // Add regularization to Hessian
      arma::mat hessian = XtWX + lambda * arma::eye(p, p);
      
      // Update beta using damped Newton step
      beta.col(k) -= eta * solve(hessian, grad);
    }
    
    // Update probabilities for next iteration
    P = softmax_matrix_c(X, beta);
    
    // Calculate objective value for this iteration
    objective(i + 1) = loss_c(y, P, beta, lambda);
  }
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("objective") = objective
  );
} 