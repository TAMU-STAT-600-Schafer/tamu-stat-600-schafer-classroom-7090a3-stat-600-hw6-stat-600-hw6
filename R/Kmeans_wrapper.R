#' Title
#' K-means s
#'
#' @param X         # n * p data matrix, n is the data size and p is the dimension
#' @param K         # K * p positions of centroids, K is the number of clusters and p is the dimension
#' @param M         # K * p initial centroids
#' @param numIter   # number of iterations
#'
#' @return Explain return
#' This function returns a vector of length n containing the indices of cluster that
#' each points have been assigned to 
#' @export
#' 
#' @examples
#' # Calling the MyKmeans function to sort the rows of a 20 x 50 matrix X
#' # into 5 clusters, using the first 5 rows of X as the initial centroids.
#' 
#' X <- matrix(rnorm(1000), 20, 50)
#' Y <- MyKmeans(X, 5, X[1:5, ], 100)
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