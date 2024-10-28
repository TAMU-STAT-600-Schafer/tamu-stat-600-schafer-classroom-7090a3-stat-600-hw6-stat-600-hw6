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
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                               int numIter = 50, double eta = 0.1, double lambda = 1){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int K = max(y) + 1; // number of classes
    int p = X.n_cols;
    int n = X.n_rows;
    std::cout << "after kpn";
    arma::mat beta = beta_init; // to store betas and be able to change them if needed
    arma::vec objective(numIter + 1); // to store objective values
    
    std::cout << "after initalization";
    
    // Initialize anything else that you may need
    // Initialize probability matrix
    arma::mat P = arma::zeros(n, K);
    
    // Calculate initial probabilities and objective value
    arma::mat XB = X * beta;
    
    XB.each_row() -= arma::max(XB, 0);  // Numerical stability
    arma::mat exp_XB = arma::exp(XB);
    arma::vec row_sums = arma::sum(exp_XB, 1);
    P = exp_XB.each_col() / row_sums;
    
    // Calculate initial objective value
    double reg_term = 0.5 * lambda * accu(beta % beta);
    objective(0) = -accu(arma::log(P.elem(arma::find(P > 0)))) + reg_term;
    std::cout << "Before ForLOOp";
    // Newton's method cycle
    for(int iter = 0; iter < numIter; iter++) {
      for(int k = 0; k < K; k++) {
        // Calculate weights
        arma::vec w = P.col(k) % (1 - P.col(k));
        
        // Create weighted X
        arma::mat X_weighted = X.each_col() % w;
        
        // Calculate XtWX
        arma::mat XtWX = X.t() * X_weighted;
        std::cout << "after XtWX";
        // Create indicator vector for current class
        arma::vec y_indicator = arma::conv_to<arma::vec>::from(y == k);
        // Calculate gradient and add regularization
        arma::vec grad = X.t() * (P.col(k) - y_indicator) + lambda * beta.col(k);
        std::cout << "after gradient step";
        // Add regularization to Hessian
        arma::mat hessian = XtWX + lambda * arma::eye(p, p);
        
        // Update beta using damped Newton step
        beta.col(k) -= eta * arma::solve(hessian, grad);
      }
      
      // Update probabilities for next iteration
      XB = X * beta;
      XB.each_row() -= arma::max(XB, 0);  // Numerical stability
      exp_XB = arma::exp(XB);
      row_sums = arma::sum(exp_XB, 1);
      P = exp_XB.each_col() / row_sums;
      
      // Calculate objective value for this iteration
      reg_term = 0.5 * lambda * accu(beta % beta);
      objective(iter + 1) = -accu(arma::log(P.elem(arma::find(P > 0)))) + reg_term;
    }
    
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
