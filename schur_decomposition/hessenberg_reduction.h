#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/householder_reflection.h"

namespace hessenberg_reduction {

template <typename Scalar>
class HessenbergReduction {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

 public:
  using DynamicMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using DynamicVector = Eigen::Matrix<Scalar, -1, 1>;
  using HouseholderReflector =
      householder_reflection::HouseholderReflector<Scalar>;

  void run(DynamicMatrix* data, DynamicMatrix* backtrace) {
    set_internal_members(data, backtrace);
    reduce_matrix();
  }

 private:
  void reduce_matrix() {
    for (int cur_col = 0; cur_col < size() - 2; ++cur_col) {
      reduce_column(cur_col);
    }
  }

  void reduce_column(int cur_col) {
    HouseholderReflector reflector = HouseholderReflector(
        p_hessenberg_form_->col(cur_col).bottomRows(size() - cur_col - 1));
    update_hessenberg_form(reflector, size() - cur_col - 1);
    update_hessenberg_form(reflector, size() - cur_col - 1);
  }

  void update_hessenberg_form(const HouseholderReflector& reflector,
                              int block_size) {
    DynamicVector first = reflector.direction();
    DynamicVector second =
        p_hessenberg_form_->bottomRightCorner(size(), block_size) * first;
    second.tail(block_size) -=
        first * (first.transpose() * second.tail(block_size));
    second *= 2;

    DynamicMatrix tmp = first * second.transpose();
    p_hessenberg_form_->bottomLeftCorner(block_size, size()) -= tmp;
    p_hessenberg_form_->topRightCorner(size(), block_size) -= tmp.transpose();
  }

  void update_backtrace(const HouseholderReflector& reflector, int block_size) {
    reflector.reflect_right(
        p_backtrace_matrix_->bottomRightCorner(size(), block_size));
  }

  void set_internal_members(DynamicMatrix* data, DynamicMatrix* backtrace) {
    assert(data);
    assert(data->rows() == data->cols());
    p_hessenberg_form_ = data;

    assert(backtrace);
    p_backtrace_matrix_ = backtrace;
    *p_backtrace_matrix_ = DynamicMatrix::Identity(size(), size());
  }

  int size() {
    assert(p_hessenberg_form_->rows() == p_hessenberg_form_->cols());
    return (p_hessenberg_form_->rows());
  }

  DynamicMatrix* p_hessenberg_form_;
  DynamicMatrix* p_backtrace_matrix_;
};

}  // namespace hessenberg_reduction
