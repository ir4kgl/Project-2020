#include "householder_reflection.h"

namespace hessenberg_reduction {

template <typename Scalar>
class Hessenberg_reduction {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using MatrixDynamic = Eigen::Matrix<Scalar, -1, -1>;
  using UnitaryMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using HouseholderReflector =
      householder_reflection::Householder_reflector<Scalar>;

 public:
  void run(MatrixDynamic* data, UnitaryMatrix* backtrace) {
    assert(data.rows() == data.cols());
    full_size_ = data->rows();
    p_hessenberg_form_ = data;
    p_backtrace_matrix_ = backtrace;
    rows_cnt_ = full_size_;

    for (col_ = 0; col_ < full_size_ - 2; ++col_) {
      --rows_cnt_;
      reflector_ =
          Reflector(p_hessenberg_form_->col(col_).bottomRows(rows_cnt_));
      reduce_column();
    }
    reflector_ = {};
  }

 private:
  void reduce_column() {
    reflector_.reflect_left(
        p_hessenberg_form_->bottomRightCorner(rows_cnt_, full_size_ - col_));
    reflector_.reflect_right(
        p_hessenberg_form_->bottomRightCorner(full_size_, rows_cnt_));
    reflector_.reflect_right(
        p_backtrace_matrix_->bottomRightCorner(full_size_, rows_cnt_));
  }

  MatrixDynamic* p_hessenberg_form_;
  UnitaryMatrix* p_backtrace_matrix_;
  HouseholderReflector reflector_;

  int full_size_;
  int col_;
  int rows_cnt_;

};  // namespace hessenberg_reduction

}  // namespace hessenberg_reduction
