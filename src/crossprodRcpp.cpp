#include <RcppEigen.h>
using namespace Rcpp;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::Lower;

// [[Rcpp::export]]
NumericMatrix crossprodnum(SEXP AA){
  const Map<MatrixXd> A(as<Map<MatrixXd> >(AA));
  const int n(A.cols());
  MatrixXd AtA(MatrixXd(n, n).setZero().selfadjointView<Lower>().rankUpdate(A.adjoint()));
  return wrap(AtA);
}

// [[Rcpp::export]]
NumericMatrix tcrossprodnum(SEXP AA){
  const Map<MatrixXd> A(as<Map<MatrixXd> >(AA));
  const int n(A.rows());
  MatrixXd AAt(MatrixXd(n, n).setZero().selfadjointView<Lower>().rankUpdate(A));
  return wrap(AAt);
}

// [[Rcpp::export]]
NumericMatrix crossprodint(SEXP AA){
  const Map<MatrixXi> A(as<Map<MatrixXi> >(AA));
  const int n(A.cols());
  MatrixXi AtA(MatrixXi(n, n).setZero().selfadjointView<Lower>().rankUpdate(A.adjoint()));
  return wrap(AtA);
}

// [[Rcpp::export]]
NumericMatrix tcrossprodint(SEXP AA){
  const Map<MatrixXi> A(as<Map<MatrixXi> >(AA));
  const int n(A.rows());
  MatrixXi AAt(MatrixXi(n, n).setZero().selfadjointView<Lower>().rankUpdate(A));
  return wrap(AAt);
}

// [[Rcpp::export]]
NumericMatrix tcrossprodnumnum(SEXP AA, SEXP BB){
  const Map<MatrixXd> A(as<Map<MatrixXd> >(AA));
  const Map<MatrixXd> B(as<Map<MatrixXd> >(BB));
  return wrap(A * B.adjoint());
}

// [[Rcpp::export]]
NumericMatrix tcrossprodintint(SEXP AA, SEXP BB){
  const Map<MatrixXi> A(as<Map<MatrixXi> >(AA));
  const Map<MatrixXi> B(as<Map<MatrixXi> >(BB));
  return wrap(A * B.adjoint());
}

// [[Rcpp::export]]
NumericMatrix crossprodnumnum(SEXP AA, SEXP BB){
  const Map<MatrixXd> A(as<Map<MatrixXd> >(AA));
  const Map<MatrixXd> B(as<Map<MatrixXd> >(BB));
  return wrap(A.adjoint() * B);
}

// [[Rcpp::export]]
NumericMatrix crossprodintint(SEXP AA, SEXP BB){
  const Map<MatrixXi> A(as<Map<MatrixXi> >(AA));
  const Map<MatrixXi> B(as<Map<MatrixXi> >(BB));
  return wrap(A.adjoint() * B);
}

// [[Rcpp::export]]
NumericMatrix crossprodIntint(SEXP AA, SEXP BB){
  const Map<MatrixXi> A(as<Map<MatrixXi> >(AA));
  const Map<VectorXi> B(as<Map<VectorXi> >(BB));
  return wrap(B*A);
}

// [[Rcpp::export]]
NumericMatrix prodnumnum(SEXP AA, SEXP BB){
  const Map<MatrixXd> A(as<Map<MatrixXd> >(AA));
  const Map<MatrixXd> B(as<Map<MatrixXd> >(BB));
  return wrap(A * B);
}

// [[Rcpp::export]]
NumericMatrix prodintint(SEXP AA, SEXP BB){
  const Map<MatrixXi> A(as<Map<MatrixXi> >(AA));
  const Map<MatrixXi> B(as<Map<MatrixXi> >(BB));
  return wrap(A * B);
}

// [[Rcpp::export]]
NumericMatrix eprodnumnum(SEXP AA, SEXP BB){
  const Map<MatrixXd> A(as<Map<MatrixXd> >(AA));
  const Map<MatrixXd> B(as<Map<MatrixXd> >(BB));
  return wrap(A.cwiseProduct(B));
}

// [[Rcpp::export]]
NumericMatrix eprodintint(SEXP AA, SEXP BB){
  const Map<MatrixXi> A(as<Map<MatrixXi> >(AA));
  const Map<MatrixXi> B(as<Map<MatrixXi> >(BB));
  return wrap(A.cwiseProduct(B));
}

// [[Rcpp::export]]
IntegerVector sseintint(SEXP AA, SEXP BB){
  const Map<MatrixXi> A(as<Map<MatrixXi> >(AA));
  const Map<MatrixXi> B(as<Map<MatrixXi> >(BB));
  MatrixXi AB = A-B;
  return wrap(AB.cwiseProduct(AB).sum());
}

// [[Rcpp::export]]
NumericVector ssenumnum(SEXP AA, SEXP BB){
  const Map<MatrixXd> A(as<Map<MatrixXd> >(AA));
  const Map<MatrixXd> B(as<Map<MatrixXd> >(BB));
  MatrixXd AB = A-B;
  return wrap(AB.cwiseProduct(AB).sum());
}
