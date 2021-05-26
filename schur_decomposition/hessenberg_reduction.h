#ifndef _SCHUR_DECOMPOSITION_HESSENBERG_REDUCTION_H
#define _SCHUR_DECOMPOSITION_HESSENBERG_REDUCTION_H

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/householder_reflection.h"

namespace hessenberg_reduction {

template <typename Scalar>
class HessenbergReduction {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

 public:
  using HouseholderReflector =
      householder_reflection::HouseholderReflector<Scalar>;
  using DynamicMatrix = HouseholderReflector::DynamicMatrix;
  using DynamicVector = HouseholderReflector::DynamicVector;

  void run(DynamicMatrix* data, DynamicMatrix* backtrace) {
    set_internal_resources(data, backtrace);
    reduce_matrix();
  }

 private:
  void reduce_matrix() {
    for (int cur_col = 0; cur_col < data_size() - 2; ++cur_col) {
      reduce_column(cur_col);
    }
  }

  void reduce_column(int cur_col) {
    HouseholderReflector reflector =
        HouseholderReflector(find_reduced_col(cur_col));
    update_hessenberg_form(data_size() - cur_col - 1, reflector);
    update_backtrace(data_size() - cur_col - 1, reflector);
  }

  DynamicVector find_reduced_col(int col) {
    return p_hessenberg_form_->col(col).bottomRows(data_size() - col - 1);
  }

  void update_hessenberg_form(int block_size,
                              const HouseholderReflector& reflector) {
    DynamicVector first = reflector.direction();
    DynamicVector second =
        p_hessenberg_form_->bottomRightCorner(data_size(), block_size) * first;
    second.tail(block_size) -=
        first * (first.transpose() * second.tail(block_size));
    second *= 2;

    DynamicMatrix tmp = first * second.transpose();
    p_hessenberg_form_->bottomLeftCorner(block_size, data_size()) -= tmp;
    p_hessenberg_form_->topRightCorner(data_size(), block_size) -=
        tmp.transpose();
  }

  void update_backtrace(int block_size, const HouseholderReflector& reflector) {
    reflector.reflect_right(
        p_backtrace_matrix_->bottomRightCorner(data_size(), block_size));
  }

  void set_internal_resources(DynamicMatrix* data, DynamicMatrix* backtrace) {
    assert(data);
    assert(data->rows() == data->cols());
    p_hessenberg_form_ = data;

    assert(backtrace);
    p_backtrace_matrix_ = backtrace;
    *p_backtrace_matrix_ = DynamicMatrix::Identity(data_size(), data_size());
  }

  int data_size() {
    assert(p_hessenberg_form_->rows() == p_hessenberg_form_->cols());
    return p_hessenberg_form_->rows();
  }

  DynamicMatrix* p_hessenberg_form_;
  DynamicMatrix* p_backtrace_matrix_;
};

}  // namespace hessenberg_reduction

#endif
