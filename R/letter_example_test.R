# Application of multi-class logistic to letters data

# Load the letter data
#########################
# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

vec_1 <- rep(1, dim(X)[1])

# [ToDo] Make sure to add column for an intercept to X and Xt
X <- cbind(vec_1, X)

Xt <- cbind(1, Xt)

library(profvis)
# Source the LR function
source("R/FunctionsLR.R")

# calling classification function and prediction
# profvis({
out <- LRMultiClass_R(X, y,numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL)
# })
# [ToDo] Try the algorithm LRMultiClass with lambda = 1 and 50 iterations. Call the resulting object out, i.e. out <- LRMultiClass(...)

# The code below will draw pictures of objective function, as well as train/test error over the iterations
plot(out$objective, type = "o")
plot(out$error_train, type = "o")
out$objective
#plot(out$error_test, type = "o")




source("R/compare_r.R")

# calling classification function and prediction
# profvis({
out2 <- LRMultiClass_com_R(X, y,numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL)
# })
# [ToDo] Try the algorithm LRMultiClass with lambda = 1 and 50 iterations. Call the resulting object out, i.e. out <- LRMultiClass(...)

# The code below will draw pictures of objective function, as well as train/test error over the iterations
plot(out2$objective, type = "o")
plot(out2$error_train, type = "o")
out2$objective



# profvis({
library(Rcpp)
source("R/LR_wrapper.R")
out1 <- LRMultiClass(X, y,numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL)
# })
# [ToDo] Try the algorithm LRMultiClass with lambda = 1 and 50 iterations. Call the resulting object out, i.e. out <- LRMultiClass(...)

# The code below will draw pictures of objective function, as well as train/test error over the iterations
plot(out1$objective, type = "o")
plot(out1$error_train, type = "o")


# Feel free to modify the code above for different lambda/eta/numIter values to see how it affects the convergence as well as train/test errors
#library(microbenchmark)
# [ToDo] Use microbenchmark to time your code with lambda=1 and 50 iterations. To save time, only apply microbenchmark 5 times.
#microbenchmark(
#  {
#    LRMultiClass(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL)
#  },
#  times = 5
#)

# [ToDo] Report the median time of your code from microbenchmark above in the comments below

# Median time: 1.54 (in sec)


####################################################################################################################################################################
# Application of multi-class logistic to letters data
# Load the letter data
#########################
# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])
# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])
# [ToDo] Make sure to add column for an intercept to X and Xt

vec_1 <- rep(1, dim(X)[1])
vec_2 <- rep(1, dim(Xt)[1])

X <- cbind(vec_1, X)
Xt <- cbind(vec_2, Xt)

# Source the LR function
source("R/FunctionsLR.R")
library(microbenchmark)
# [ToDo] Try the algorithm LRMultiClass with lambda = 1 and 50 iterations. Call the resulting object out, i.e. out <- LRMultiClass(...)

out <- LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL)
sourceCpp("src/")
out2 <- LRMultiClass_2(X, Y, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL)

microbenchmark(
  LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL),
  LRMultiClass_2(X, Y, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL),
  times = 10
)

print(out)
print(out2)

# The code below will draw pictures of objective function, as well as train/test error over the iterations
plot(out$objective, type = 'o', main = "Objective Function", xlab = "Iteration", ylab = "Objective Value")
plot(out$error_train, type = 'o')
plot(out$error_test, type = 'o')
