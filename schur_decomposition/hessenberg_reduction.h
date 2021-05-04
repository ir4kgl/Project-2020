#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/householder_reflection.h"

namespace hessenberg_reduction {

using householder_reflection::HouseholderReflector;
using std::is_arithmetic_v;

template <typename Scalar>
class HessenbergReduction {
  static_assert(is_arithmetic_v<Scalar>, "Scalar must be arithmetic type!");

 public:
  using SquareMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using UnitaryMatrix = Eigen::Matrix<Scalar, -1, -1>;

  void run(SquareMatrix* data, UnitaryMatrix* backtrace) {
    assert(data->rows() == data->cols());
    data_size_ = data->rows();
    p_hessenberg_form_ = data;
    p_backtrace_matrix_ = backtrace;

    *p_backtrace_matrix_ = UnitaryMatrix::Identity(data_size_, data_size_);
    for (cur_col_ = 0; cur_col_ < data_size_ - 2; ++cur_col_) {
      cur_block_size_ = data_size_ - cur_col_ - 1;
      reflector_ = HouseholderReflector<Scalar>(
          p_hessenberg_form_->col(cur_col_).bottomRows(cur_block_size_));
      reduce_column();
    }
    reflector_ = {};
  }

 private:
  void reduce_column() {
    reflector_.reflect_left(p_hessenberg_form_->bottomRightCorner(
        cur_block_size_, data_size_ - cur_col_));
    reflector_.reflect_right(
        p_hessenberg_form_->bottomRightCorner(data_size_, cur_block_size_));
    reflector_.reflect_right(
        p_backtrace_matrix_->bottomRightCorner(data_size_, cur_block_size_));
  }

  SquareMatrix* p_hessenberg_form_;
  UnitaryMatrix* p_backtrace_matrix_;
  HouseholderReflector<Scalar> reflector_;

  int data_size_;
  int cur_col_;
  int cur_block_size_;

};  // namespace hessenberg_reduction

}  // namespace hessenberg_reduction
