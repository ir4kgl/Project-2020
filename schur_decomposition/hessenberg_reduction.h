#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/householder_reflection.h"

namespace hessenberg_reduction {

using householder_reflection::HouseholderReflector;
using std::is_arithmetic_v;

template <typename Scalar>
class HessenbergReduction {
  static_assert(is_arithmetic_v<Scalar>, "Scalar must be arithmetic type!");

  using SquareMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using UnitaryMatrix = Eigen::Matrix<Scalar, -1, -1>;

 public:
  void run(SquareMatrix* data, UnitaryMatrix* backtrace) {
    assert(data->rows() == data->cols());
    size_ = data->rows();
    p_hessenberg_form_ = data;
    p_backtrace_matrix_ = backtrace;

    *p_backtrace_matrix_ = UnitaryMatrix::Identity(size_, size_);
    for (col_ = 0; col_ < size_ - 2; ++col_) {
      zeroed_counter_ = size_ - col_ - 1;
      reflector_ = HouseholderReflector<Scalar>(
          p_hessenberg_form_->col(col_).bottomRows(zeroed_counter_));
      reduce_column();
    }
    reflector_ = {};
  }

 private:
  void reduce_column() {
    reflector_.reflect_left(
        p_hessenberg_form_->bottomRightCorner(zeroed_counter_, size_ - col_));
    reflector_.reflect_right(
        p_hessenberg_form_->bottomRightCorner(size_, zeroed_counter_));
    reflector_.reflect_right(
        p_backtrace_matrix_->bottomRightCorner(size_, zeroed_counter_));
  }

  SquareMatrix* p_hessenberg_form_;
  UnitaryMatrix* p_backtrace_matrix_;
  HouseholderReflector<Scalar> reflector_;

  int size_;
  int col_;
  int zeroed_counter_;

};  // namespace hessenberg_reduction

}  // namespace hessenberg_reduction
