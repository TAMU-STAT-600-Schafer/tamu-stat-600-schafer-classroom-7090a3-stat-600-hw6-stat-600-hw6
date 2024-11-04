#' Title
#' K-means s
#'
#' @param X 
#' @param K 
#' @param M 
#' @param numIter 
#'
#' @return Explain return
#' @export
#'
#' @examples
#' # Give example
MyKmeans <- function(X, K, M = NULL, numIter = 100){
  
  n = nrow(X) # number of rows in X
  M1 <- matrix(0, K, dim(X)[2], byrow=TRUE)
  X1 <- matrix(0, n, dim(X)[2])
  X1 <- X1 + as.matrix(X)

  # Check whether M is NULL or not. If NULL, initialize based on K random points from X. If not NULL, check for compatibility with X dimensions.
  
  if(is.null(M)){
    
    # Check that K is equal to or greater than the number of rows of X.
    if( K <= dim(X)[1]){}
    else{
      stop("Error: K must be equal to or greater than the number of rows of X.")
    }
    
    # Select K random rows from X to create the initial centroid matrix M1
    indices <- as.vector(sample(1:dim(X)[1], K))
    M1 <- M1 + X1[indices, ]
    
  }
  
  # If not NULL, check for compatibility with X dimensions and K.
  
  else{
    
    # Check for compatible dimensions.
    if( dim(X)[2] == dim(M)[2]){}
    else{
      stop("Error: X array and M array must have same number of columns.")
    }
    if( dim(M)[1] == K ){}
    else{
      stop("Error: M must have K rows.")
    }
    if( K <= dim(X)[1]){}
    else{
      stop("Error: K must be equal to or greater than the number of rows of X.")
    }
    
    # Set values of M1 equal to M
    M1 <- M1 + as.matrix(M)
    
  }
  
  # Call C++ MyKmeans_c function to implement the algorithm
  Y = MyKmeans_c(X1, K, M1, numIter)
  
  # Return the class assignments
  return(Y)
}