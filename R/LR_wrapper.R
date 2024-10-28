
#' Title
#'
#' @param X 
#' @param y 
#' @param numIter 
#' @param eta 
#' @param lambda 
#' @param beta_init 
#'
#' @return
#' @export
#'
#' @examples
#' # Give example
LRMultiClass <- function(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  
  # Compatibility checks from HW3 and initialization of beta_init
  if (!all(X[, 1] == 1)) {
    stop("The first column of X must be 1s.")
  }
  
  # Check for compatibility of dimensions between X and Y
  if (nrow(X) != length(y)) {
    stop("X number of rows doesn't match length of Y")
  }

  # Check eta is positive
  if (eta <= 0) {
    stop("Eta is not positive")
  }
  
  # Check lambda is non-negative
  if (lambda < 0) {
    stop("Lambda is negative")
  }
  
  n <- nrow(X)
  p <- ncol(X)
  K <- length(unique(y))
  
  
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (is.null(beta_init)) {
    beta_init <- matrix(0, nrow = p, ncol = K)
  } else {
    if (nrow(beta_init) != ncol(X)) {
      stop("Length of beta doesn't match number of columns of X")
    }
    if (ncol(beta_init) != K) {
      stop("Numbers of categories of beta and Y don't match")
    }
  }
  
  sourceCpp("src/LRMultiClass.cpp")
  # Call C++ LRMultiClass_c function to implement the algorithm
  print("After source")
  print(dim(X))
  print(length(y))
  print(dim(beta_init))
  out = LRMultiClass_c(X, y, beta_init, numIter, eta, lambda)
  print("After c code")
  # Return the class assignments
  return(out)
}







