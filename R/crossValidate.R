#' Cross-validation for NMF
#' 
#' @description Find an "optimal" rank for a Non-Negative Matrix Factorization using cross-validation. Returns a \code{data.frame} with class \code{nmfCrossValidate}. Plot results using the \code{plot} class method.
#' 
#' @details 
#' Two algorithms are available for cross-validation and can be selected using the \code{method} parameter.
#'
#' 1. **Bi-cross-validation**: Data is split into an equally-sized 2x2 grid (A1, A2, B1, and B2). Data is trained on A1 and projected to A2 to B2 where the mean squared error of the model on B2 is measured.
#' 2. **Cost of bipartite matching** between independent replicates. Samples are partitioned into two sets and independently factorized, then correlation between factors in both \code{w} models is computed, factors are matched using bipartite matching on a cosine similarity cost matrix, and the mean cost of matched factors is returned.
#' 
#' @inheritParams nmf
#' @param k array of factorization ranks to test
#' @param method algorithm to use for cross-validation: "\code{predict}" = bi-cross-validation on equally sized feature/sample subsets, "\code{impute}" = MSE of missing value imputation.
#' @param reps number of independent replicates to run
#' @param n for \code{method = "impute"} and \code{"perturb"}, fraction of values to handle as missing
#' @param fn optional function to apply when computing MSE on the results
#' @param ... parameters to \code{RcppML::nmf}, not including \code{data} or \code{k}
#' @return \code{data.frame} with class \code{nmfCrossValidate} with columns \code{rep}, \code{k}, and \code{value}
#' @md
#' @seealso \code{\link{nmf}}
#' @export
crossValidate <- function(data, k, method = "impute", reps = 5, n = 0.2, fn = NULL, ...) {
  verbose <- getOption("RcppML.verbose")
  options("RcppML.verbose" = FALSE)
  p <- list(...)
  defaults <- list("mask" = NULL)
  for (i in 1:length(defaults))
    if (is.null(p[[names(defaults)[[i]]]])) p[[names(defaults)[[i]]]] <- defaults[[i]]
  
  if(!is.null(p$mask)){
    if(!is.character(p$mask)){
      if(canCoerce(p$mask, "ngCMatrix")){
        p$mask <- as(p$mask, "ngCMatrix")
      } else if(canCoerce(p$mask, "matrix")){
        p$mask <- as(as.matrix(p$mask), "ngCMatrix")
      } else stop("supplied masking value was invalid or not coercible to a matrix")
    }
  }
  
  results <- list()
  for(rep in 1:reps){
    if(verbose) cat("\nReplicate ", rep, ", rank: ", sep = "")
    
    # get samples, features, or missing values for test/training/cross-validation
    if(!is.null(p$seed)) set.seed(p$seed + rep)
    mask_matrix <- rsparsematrix(as.numeric(nrow(data)), as.numeric(ncol(data)), n, rand.x = NULL)

    for(rank in k){
      if(verbose) cat(rank, " ")
      
      mask_22 <- mask_21 <- mask_11 <- mask_w1 <- mask_w2 <- p$mask
      if(method == "predict"){
        if(is(p$mask, "ngCMatrix")){
          mask_22 <- p$mask[-features, -samples]
          mask_21 <- p$mask[-features, samples]
          mask_11 <- p$mask[features, samples]
        }
        m22 <- nmf(data[-features, -samples], rank, mask = mask_22, ...)
        m21 <- predict.nmf(m22$w, data[-features, samples], mask = mask_21)
        m11 <- predict.nmf(m21, t(data[features, samples]), mask = mask_11)
        m <- new("nmf", "w" = t(m11), "d" = rep(1, rank), "h" = m21)
        value <- evaluate(m, data[features, samples], mask = mask_11)
      } else if(method == "robust"){
        if(is(p$mask, "ngCMatrix")){
          mask_w1 <- p$mask[, samples]
          mask_w2 <- p$mask[, -samples]
        }
        w1 <- nmf(data[, samples], rank, mask = mask_w1, ...)$w
        w2 <- nmf(data[, -samples], rank, mask = mask_w2, ...)$w
        value <- bipartiteMatch(1 - cosine(w1, w2))$cost / rank
      } else if(method == "impute"){
        # usually a sensible default 
        m <- nmf(data, rank, mask = mask_matrix, ...)
        value <- evaluate(m, data, mask = mask_matrix, missing_only = TRUE)
        if (!is.null(fn)) {
          fn_value <- .fn_mse(data, m, fn)
        } else {
          fn_value <- NA_integer_
        }
      } else if(method == "perturb") {
        m <- nmf(data, rank, ...)
        data@x[ind] <- ind_vals
        value <- evaluate(m, data, mask = mask_matrix, missing_only = TRUE)
        if(p$perturb_to == "random"){
          data@x[ind] <- perturb_vals
        } else {
          data@x[ind] <- 0
        }
      }
      if (!is.null(fn)) {
        results[[length(results) + 1]] <- c(rep, rank, value)
      } else {
        results[[length(results) + 1]] <- c(rep, rank, value, fn_value)
      }
    }
  }
  results <- data.frame(do.call(rbind, results))
  if (!is.null(fn)) { 
    colnames(results) <- c("rep", "k", "value", "fn_value")
  } else { 
    colnames(results) <- c("rep", "k", "value")
  }
  class(results) <- c("nmfCrossValidate", "data.frame")
  results$rep <- as.factor(results$rep)
  options("RcppML.verbose" = verbose)
  return(results)
}


# helper fn for transformed MSE if requested
.fn_mse <- function(data, m, fn) .mse(fn(data), fn(m@w %*% m@h))


# helper fn for transformed MSE if requested
.mse <- function(X, Xi) mean(na.rm=TRUE, (X - Xi)**2)
