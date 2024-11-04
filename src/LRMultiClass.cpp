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
  
  arma::mat Z = X * beta;           // Initialize product of X and beta
  
  arma::mat Z_exp = exp(Z);         // Initialize Z_exp matrix whose indices Z_exp(i,j) = exp(X*beta(i,j))
  arma::vec Z_sum = sum(Z_exp, 1);  // Sum the rows of Z_exp
  
  return Z_exp.each_col() / Z_sum;  // Return Z_exp with each column normalized
}

// [[Rcpp::export]]
double loss_c(const arma::uvec& y, const arma::mat& P, const arma::mat& beta, const double lambda) {
  
  // Initialize sum variable
  double sum1 = 0;
  
  for(int k = 0; k < P.n_cols; k++){
    
    // Find indices of y where y(i) = k
    arma::uvec y_index = arma::find(y == k);
    
    // Sum all log(P(i, k)) such that y(i) = k
    for(int i = 0; i < y_index.n_elem; i++){
      sum1 += std::log(P(y_index(i), k));
    }
  }
  
  return -sum1 + arma::dot(beta, beta) * (lambda / 2);  // Return difference of weighted F2 norm of beta and the sum of log(P(y==k,k))
}


// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                          int numIter = 50, double eta = 0.1, double lambda = 1) {
  int n = X.n_rows;
  int p = X.n_cols;
  int K = arma::max(y) + 1;  // number of classes
  
  // Initialize variables
  arma::mat Xt = X.t();                               // X transpose        
  arma::mat X_weighted = X;                           // Matrix to compute weighted X    
  arma::mat beta = beta_init;                         // Copy initial beta matrix    
  arma::vec objective(numIter + 1);                   // Record initial objective value
  arma::vec lambda_vec(p, arma::fill::zeros);         // Create lambda vector      
  lambda_vec += lambda;                                         
  
  // Calculate initial probabilities
  arma::mat P = softmax_matrix_c(X, beta_init);
  
  // Calculate initial objective value
  objective(0) = loss_c(y, P, beta_init, lambda);
  
  // Newton's method cycle
  for(int i = 0; i < numIter; i++) {
    for(int k = 0; k < K; k++) {
      
      // Calculate weights
      arma::vec w = P.col(k) % (1 - P.col(k));
      
      // Reset weighted X (matching R's sweep operation)
      X_weighted = X;
      
      // Multiply each row by corresponding weight
      for(int j = 0; j < n; j++){
        X_weighted.row(j) *= w(j);
      }
      
      // Calculate X'*diag(weights)*X
      arma::mat hessian = Xt * X_weighted;
      
      // Create indicator vector for current class (k-1 to match R indexing)
      arma::vec y_indicator = arma::conv_to<arma::vec>::from(y == k);
      
      // Calculate gradient and add regularization
      arma::vec grad = Xt * (P.col(k) - y_indicator) + lambda * beta.col(k);
      
      // Add regularization to Hessian
      hessian += arma::diagmat(lambda_vec);
      
      // Update beta using damped Newton step
      beta.col(k) -= eta * solve(hessian, grad);
    }
    
    // Update probabilities for next iteration
    P = softmax_matrix_c(X, beta);
    
    // Calculate objective value for this iteration
    objective(i + 1) = loss_c(y, P, beta, lambda);
  }
  
  // Return list of final beta matrix and vector of objective function values from each iteration
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("objective") = objective
  );
}