#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/householder_reflection.h"
#include "../schur_decomposition/tridiagonal_symmetric.h"

namespace hessenberg_reduction {

using std::is_arithmetic_v;

template <typename Scalar>
class HessenbergReduction {
  static_assert(is_arithmetic_v<Scalar>, "Scalar must be arithmetic type!");

 public:
  using DynamicMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using DynamicVector = Eigen::Matrix<Scalar, -1, 1>;

  using HouseholderReflector =
      householder_reflection::HouseholderReflector<Scalar>;
  using TridiagonalSymmetric =
      tridiagonal_symmetric::TridiagonalSymmetric<Scalar>;

  void run(const DynamicMatrix& data,
           TridiagonalSymmetric* hessenberg_form_diagonals,
           DynamicMatrix* backtrace) {
    assert(data.rows() == data.cols());
    assert(hessenberg_form_diagonals);
    assert(backtrace);

    hessenberg_form_ = data;
    p_hessenberg_form_diagonals_ = hessenberg_form_diagonals;
    p_backtrace_matrix_ = backtrace;

    data_size_ = data.rows();
    p_hessenberg_form_diagonals_->set_size(data_size_);
    *p_backtrace_matrix_ = DynamicMatrix::Identity(data_size_, data_size_);

    reduce_matrix();
  }

 private:
  void reduce_matrix() {
    for (int cur_col = 0; cur_col < data_size_ - 2; ++cur_col) {
      int cur_block_size = data_size_ - cur_col - 1;
      reduce_column(cur_col, cur_block_size);
    }

    extract_diagonals();
    reflector_ = {};
    hessenberg_form_ = {};
  }

  void reduce_column(int cur_col, int cur_block_size) {
    reflector_ = HouseholderReflector(
        hessenberg_form_.col(cur_col).bottomRows(cur_block_size));

    reflect_hessenberg_form(cur_block_size);

    reflector_.reflect_right(
        p_backtrace_matrix_->bottomRightCorner(data_size_, cur_block_size));
  }

  void reflect_hessenberg_form(int block_size) {
    DynamicVector first = reflector_.direction();
    DynamicVector second =
        hessenberg_form_.bottomRightCorner(data_size_, block_size) * first;
    second.tail(block_size) -=
        first * (first.transpose() * second.tail(block_size));
    second *= 2;

    DynamicMatrix tmp = first * second.transpose();
    hessenberg_form_.bottomLeftCorner(block_size, data_size_) -= tmp;
    hessenberg_form_.topRightCorner(data_size_, block_size) -= tmp.transpose();
  }

  void extract_diagonals() {
    p_hessenberg_form_diagonals_->set_major_diagonal(
        hessenberg_form_.diagonal(0));
    p_hessenberg_form_diagonals_->set_side_diagonal(
        hessenberg_form_.diagonal(1));
  }

  DynamicMatrix* p_backtrace_matrix_;
  TridiagonalSymmetric* p_hessenberg_form_diagonals_;

  DynamicMatrix hessenberg_form_;
  HouseholderReflector reflector_;
  int data_size_;
};

}  // namespace hessenberg_reduction
