# Create a function to sum the log values of the indices of the probability matrix that correspond to the value of Y.
function_beta <- function(i, Pk_mat, y){
  indicator_sum <- sum(log(Pk_mat[y == (i-1), i]))
}

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  
  # Initialize dimension variables
  n1 <- dim(X)[1]
  #n2 <- dim(Xt)[1]
  p <- dim(X)[2]
  K <- length(unique(y))
  
  # Store the transpose of X for quicker calculation in the main algorithm
  X_trans <- t(X)
  
  # Initialize vectors to store the values of the objective, test error, and training error throughout the algorithm
  objective <- rep(0, numIter+1)
  error_train <- rep(0, numIter+1)
  error_test <- rep(0, numIter+1)
  
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  
  #if(sum(as.numeric(X[ ,1] == 1)) == n1 && sum(as.numeric(Xt[ ,1] == 1)) == n2){}
  #else{
  #  stop("Error: First column of X and Xt must be 1's.")
  #}
  
  # Check for compatibility of dimensions between X and Y
  
  if(length(y) == n1){}
  else{
    stop("Error: incompatible dimensions with X and Y.")
  }
  
  # Check for compatibility of dimensions between Xt and Yt
  
  #if(length(yt) == n2){}
  #else{
  #  stop("Error: incompatible dimensions with Xt and Yt.")
  #}
  
  # Check for compatibility of dimensions between X and Xt
  
  if(dim(X)[2] == dim(Xt)[2]){}
  else{
    stop("Error: incompatible dimensions with X and Xt")
  }
  
  # Check eta is positive
  
  if(eta <= 0){
    stop("Error: eta must be positive.")
  }
  
  # Check lambda is non-negative
  
  if(lambda < 0){
    stop("Error: lambda must be non-negative")
  }
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  
  if(is.null(beta_init)){
    beta <- matrix(0, nrow = dim(X)[2], ncol = K)
  }
  else{
    beta <- beta_init
  }
  
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  
  # Generate a probability matrix for the training data
  X_beta <- X %*% beta
  exp_X_beta <- exp(X_beta)
  Pk_mat <- exp_X_beta/rowSums(exp_X_beta)
  
  # Calculate and store the initial value of the objective function
  sum_log <- sapply(1:K, \(i){function_beta(i, Pk_mat, y)})
  objective[1] <- sum(beta * beta) * (lambda / 2) - sum(sum_log)
  
  # Calculate and store the initial training error
  y_guess <- max.col(Pk_mat)-1
  error_train[1] <- (1 - sum(as.numeric(y == y_guess))/n1) * 100
  
  # Generate a probability matrix for the testing data # Calculate the first testing error
  #Xt_beta <- Xt %*% beta
  #exp_Xt_beta <- exp(Xt_beta)
  #Pk_test <- exp_Xt_beta/rowSums(exp_Xt_beta)
  
  # Calculate and store the initial testing error
  #yt_guess <- max.col(Pk_test)-1
  #error_test[1] <- (1 - sum(as.numeric(yt == yt_guess))/n2) * 100
  
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
  
  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  
  for(i in 1:numIter){
    
    # Run a loop to update the columns of beta
    for(k in 1:K){
      
      # Compute the Hessian
      w <- Pk_mat[ , k] * (1-Pk_mat[ , k])
      hessian <- X_trans %*% (w * X)
      diag(hessian) <- diag(hessian) + lambda
      #hessian_inv <- chol2inv(chol(hessian))
      hessian_inv <- solve(hessian)
      
      # Compute the Jacobian
      prob_vec <- Pk_mat[ , k] - as.numeric(y == (k-1))
      jacobian <- X_trans %*% prob_vec + lambda * beta[ , k]
      
      # Update beta
      beta[ , k] <- beta[ , k] - eta * hessian_inv %*% jacobian
      
    }
    
    # Update the probability matrix
    X_beta <- X %*% beta
    exp_X_beta <- exp(X_beta)
    Pk_mat <- exp_X_beta/rowSums(exp_X_beta)
    
    # Calculate and store the training error for iteration i
    y_guess <- max.col(Pk_mat)-1
    error_train[i+1] <- (1 - sum(as.numeric(y == y_guess))/n1) * 100
    
    # Calculate and store the objective funciton value for iteration i
    sum_log <- sapply(1:K, \(i){function_beta(i, Pk_mat, y)})
    objective[i+1] <- sum(beta * beta) * (lambda / 2) - sum(sum_log)
    
    # Update the probability matrix for the test data
    Xt_beta <- Xt %*% beta
    exp_Xt_beta <- exp(Xt_beta)
    Pk_test <- exp_Xt_beta/rowSums(exp_Xt_beta)
    
    #Calculate and store the testing error for iteration i
    yt_guess <- max.col(Pk_test)-1
    error_test[i+1] <- (1 - sum(as.numeric(yt == yt_guess))/n2) * 100
  }
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta, error_train = error_train, error_test = error_test, objective =  objective))
}