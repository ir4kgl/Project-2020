#ifndef _SCHUR_DECOMPOSITION_HESSENBERG_REDUCTION_H
#define _SCHUR_DECOMPOSITION_HESSENBERG_REDUCTION_H

#include "../eigen/Eigen/Dense"
#include "householder_reflection.h"

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
    update_hessenberg(cur_col, reflector);
    update_backtrace(cur_col, reflector);
  }

  DynamicVector find_reduced_col(int cur_col) {
    return p_hessenberg_form_->block(cur_col + 1, cur_col,
                                     data_size() - cur_col - 1, 1);
  }

  void update_hessenberg(int cur_col, const HouseholderReflector& reflector) {
    reflector.reflect_left(p_hessenberg_form_->bottomRightCorner(
        data_size() - cur_col - 1, data_size() - cur_col));
    reflector.reflect_right(p_hessenberg_form_->bottomRightCorner(
        data_size(), data_size() - cur_col - 1));
  }

  void update_backtrace(int cur_col, const HouseholderReflector& reflector) {
    reflector.reflect_right(p_backtrace_matrix_->bottomRightCorner(
        data_size(), data_size() - cur_col - 1));
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
