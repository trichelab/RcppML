# Rcpp Machine Learning Library

[![](https://cranlogs.r-pkg.org/badges/grand-total/RcppML)](https://cran.r-project.org/package=RcppML)
[![](https://www.r-pkg.org/badges/version-last-release/RcppML)](https://cran.r-project.org/package=RcppML)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

`RcppML` is an R package for fast **non-negative matrix factorization** and **divisive clustering** using **large sparse matrices**. 

RcppML NMF is:
1. **more interpretable and robust** due to diagonal scaling
2. The **fastest** NMF implementation

## Installation

Install from [CRAN](https://cran.r-project.org/web/packages/RcppML/index.html) or the development version from GitHub:

```
install.packages('RcppML')                   # install CRAN version
devtools::install_github("zdebruine/RcppML") # compile dev version
```

Check out the [CRAN manual](https://cran.r-project.org/web/packages/RcppML/RcppML.pdf).

Once installed and loaded, RcppML C++ headers defining classes can be loaded can be used in any R package using `#include <RcppML.hpp>`. 

## Matrix Factorization
Sparse matrix factorization by alternating least squares:
* Non-negativity constraints
* L1 regularization
* Diagonal scaling
* Rank-1 and Rank-2 specializations (~2x faster than _irlba_ SVD equivalents)

Read (and cite) our [bioRXiv manuscript](https://www.biorxiv.org/content/10.1101/2021.09.01.458620v1) on NMF for single-cell experiments.

#### R functions
The `nmf` function runs matrix factorization by alternating least squares in the form `A = WDH`. The `project` function updates `w` or `h` given the other, while the `mse` function calculates mean squared error of the factor model.

```{R}
A <- Matrix::rsparsematrix(A, 1000, 1000, 0.1) # sparse Matrix::dgCMatrix
model <- RcppML::nmf(A, k = 10, nonneg = TRUE)
h0 <- RcppML::project(A, w = model$w)
RcppML::mse(A, m$w, m$d, m$h)
```

#### C++ class
The `RcppML::MatrixFactorization` class is an object-oriented interface with methods for fitting, projecting, and evaluating linear factor models. It also contains a sparse matrix class equivalent to `Matrix::dgCMatrix` in R.

```{Rcpp}
#include <RcppML.hpp>

//[[Rcpp::export]]
Rcpp::List RunNMF(const Rcpp::S4& A_, int k){
     RcppML::SparseMatrix A(A_); // zero-copy, unlike arma or Eigen equivalents
     RcppML::MatrixFactorization model(k, A.rows(), A.cols());
     model.tol = 1e-5;
     model.fit(A);
     return Rcpp::List::create(
          Rcpp::Named("w") = model.w,
          Rcpp::Named("d") = model.d,
          Rcpp::Named("h") = model.h,
          Rcpp::Named("mse") = model.mse(A));
}
```

## Divisive Clustering
Divisive clustering by rank-2 spectral bipartitioning.
* 2nd SVD vector is linearly related to the difference between factors in rank-2 matrix factorization.
* Rank-2 matrix factorization (optional non-negativity constraints) for spectral bipartitioning **~2x faster** than _irlba_ SVD
* Sensitive distance-based stopping criteria similar to Newman-Girvan modularity, but orders of magnitude faster
* Stopping criteria based on minimum number of samples

#### R functions
The `dclust` function runs divisive clustering by recursive spectral bipartitioning, while the `bipartition` function exposes the rank-2 NMF specialization and returns statistics of the bipartition.

```{R}
A <- Matrix::rsparsematrix(A, 1000, 1000, 0.1) # sparse Matrix::dgcMatrix
clusters <- dclust(A, min_dist = 0.001, min_samples = 5)
cluster0 <- bipartition(A)
```

#### C++ class
The `RcppML::clusterModel` class provides an interface to divisive clustering. In the future, more clustering algorithms may be added.

```{Rcpp}
#include <RcppML.hpp>

//[[Rcpp::export]]
Rcpp::List DivisiveCluster(const Rcpp::S4& A_, int min_samples, double min_dist){
   RcppML::SparseMatrix A(A_);
   RcppML::clusterModel model(A, min_samples, min_dist);
   model.dclust();
   std::vector<RcppML::cluster> clusters = m.getClusters();
   Rcpp::List result(clusters.size());
   for (int i = 0; i < clusters.size(); ++i) {
        result[i] = Rcpp::List::create(
             Rcpp::Named("id") = clusters[i].id,
             Rcpp::Named("samples") = clusters[i].samples,
             Rcpp::Named("center") = clusters[i].center);
   }
   return result;
}
```
