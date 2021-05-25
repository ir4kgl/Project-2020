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

  void run(DynamicMatrix* data, DynamicMatrix* backtrace) {
    set_internal_members(data, backtrace);
    reduce_matrix();
  }

 private:
  void reduce_matrix() {
    for (int cur_col = 0; cur_col < data_size() - 2; ++cur_col) {
      reduce_column(cur_col);
    }
  }

  void reduce_column(int cur_col) {
    int cur_block_size = data_size() - cur_col - 1;
    HouseholderReflector reflector = HouseholderReflector(
        p_hessenberg_form_->col(cur_col).bottomRows(cur_block_size));
    reflector.reflect_left(p_hessenberg_form_->bottomRightCorner(
        cur_block_size, data_size() - cur_col));
    reflector.reflect_right(
        p_hessenberg_form_->bottomRightCorner(data_size(), cur_block_size));
    reflector.reflect_right(
        p_backtrace_matrix_->bottomRightCorner(data_size(), cur_block_size));
  }

  void set_internal_members(DynamicMatrix* data, DynamicMatrix* backtrace) {
    assert(data->rows() == data->cols());
    p_hessenberg_form_ = data;
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
