#' L. R. Multiclass
#'
#' @param X         // Training data matrix
#' @param y         // Training vector of classifications
#' @param numIter   // Number of training iterations
#' @param eta       // Convergence tolerance
#' @param lambda    // Scalar for Frobenius norm penalty
#' @param beta_init // Initial vector of beta values
#'

#' @returns returns a coefficient matrix beta, and list of objective values from each iteration.
#' @export
#'
#' @examples
#' # Calling the LRMultiClass function on a 10 x 101 Matrix X, with first column all ones,
#' # and a vector of class assignments Y
#' 
#' X <- matrix(rnorm(1000), 10, 100)
#' X <- cbind(rep(1, 10), X)
#' y <- c(0, 1, 2, 3, 1, 2, 3, 1, 2, 3)
#' 
#' out <- LRMultiClass(X, y, 50, .1, 1, NULL)
#' 
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
  
  # Initialize dimension parameters
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

  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, beta_init, numIter, eta, lambda)
  # Return list of final beta matrix and vector of objective function values from each iteration
  return(out)
}







