#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/householder_reflection.h"

namespace hessenberg_reduction {

using householder_reflection::HouseholderReflector;
using std::is_arithmetic_v;

template <typename Scalar>
class HessenbergReduction {
  static_assert(is_arithmetic_v<Scalar>, "Scalar must be arithmetic type!");

 public:
  using DynamicMatrix = Eigen::Matrix<Scalar, -1, -1>;

  void run(DynamicMatrix* data, DynamicMatrix* backtrace) {
    assert(data->rows() == data->cols());

    data_size_ = data->rows();
    p_hessenberg_form_ = data;
    p_backtrace_matrix_ = backtrace;
    *p_backtrace_matrix_ = DynamicMatrix::Identity(data_size_, data_size_);
    reduce_matrix();
  }

 private:
  void reduce_matrix() {
    for (int cur_col = 0; cur_col < data_size_ - 2; ++cur_col) {
      int cur_block_size = data_size_ - cur_col - 1;
      reduce_column(cur_col, cur_block_size);
    }
    reflector_ = {};
  }

  void reduce_column(int cur_col, int cur_block_size) {
    reflector_ = HouseholderReflector<Scalar>(
        p_hessenberg_form_->col(cur_col).bottomRows(cur_block_size));
    reflector_.reflect_left(p_hessenberg_form_->bottomRightCorner(
        cur_block_size, data_size_ - cur_col));
    reflector_.reflect_right(
        p_hessenberg_form_->bottomRightCorner(data_size_, cur_block_size));
    reflector_.reflect_right(
        p_backtrace_matrix_->bottomRightCorner(data_size_, cur_block_size));
  }

  DynamicMatrix* p_hessenberg_form_;
  DynamicMatrix* p_backtrace_matrix_;
  HouseholderReflector<Scalar> reflector_;

  int data_size_;
};

}  // namespace hessenberg_reduction
