
#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"
#include "../schur_decomposition/hessenberg_reduction.h"

namespace schur_decomposition {

using std::abs;
using std::is_arithmetic_v;
using std::min;

template <typename Scalar>
class SchurDecomposition {
  static_assert(is_arithmetic_v<Scalar>, "Scalar must be arithmetic type!");

 public:
  using BlockDynamic = Eigen::Block<Eigen::Matrix<Scalar, -1, -1>>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

  using Rotator = givens_rotation::GivensRotator<Scalar>;
  using Reflector = householder_reflection::HouseholderReflector<Scalar>;
  using HessenbergReduction = hessenberg_reduction::HessenbergReduction<Scalar>;

  using SquareMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using UnitaryMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using SchurForm = Eigen::Matrix<Scalar, -1, -1>;
  using Precision = Scalar;

  SchurDecomposition(Precision precision) : precision_(precision) {
    assert(precision >= 0);
  }

  void run(const SquareMatrix& data, SchurForm* schur_form,
           UnitaryMatrix* unitary) {
    assert(data.rows() == data.cols());
    assert(schur_form);
    assert(unitary);
    data_size_ = data.rows();
    p_schur_form_ = schur_form;
    p_unitary_ = unitary;

    *p_schur_form_ = data;
    HessenbergReduction reduction_to_hessenberg_form;
    reduction_to_hessenberg_form.run(p_schur_form_, p_unitary_);
    for (int cur_size = data_size_ - 1; cur_size >= 2;) {
      start(cur_size);
      process(cur_size);
      finish(cur_size);
      deflate(&cur_size);
    }
  }

  void set_precision(Precision precision) {
    assert(precision >= 0);
    precision_ = precision;
  }

  Precision get_precision() const { return precision_; }

 private:
  void start(int cur_size) {
    reflector_ = Reflector(find_matching_column(cur_size));
    reflector_.reflect_left(p_schur_form_->block(0, 0, 3, data_size_));
    reflector_.reflect_right(
        p_schur_form_->block(0, 0, min(cur_size, 3) + 1, 3));
    reflector_.reflect_right(p_unitary_->block(0, 0, data_size_, 3));
  }

  void process(int cur_size) {
    int step = 0;
    for (; step <= cur_size - 3; ++step) {
      reflector_ = Reflector(p_schur_form_->block(step + 1, step, 3, 1));
      reflector_.reflect_left(
          p_schur_form_->block(step + 1, step, 3, data_size_ - step));
      reflector_.reflect_right(
          p_schur_form_->block(0, step + 1, min(cur_size, step + 4) + 1, 3));
      reflector_.reflect_right(p_unitary_->block(0, step + 1, data_size_, 3));
    }
  }

  void finish(int cur_size) {
    reflector_ =
        Reflector(p_schur_form_->block(cur_size - 1, cur_size - 2, 2, 1));
    reflector_.reflect_left(p_schur_form_->block(cur_size - 1, cur_size - 2, 2,
                                                 data_size_ - cur_size + 2));
    reflector_.reflect_right(
        p_schur_form_->block(0, cur_size - 1, cur_size + 1, 2));
    reflector_.reflect_right(p_unitary_->block(0, cur_size - 1, data_size_, 2));

    reflector_ = {};
  }

  void deflate(int* cur_size) {
    Scalar* p_corner_item = &(*p_schur_form_)(*cur_size, *cur_size - 1);
    if (near_zero(*p_corner_item)) {
      *p_corner_item = 0;
      *cur_size -= 1;
      return;
    }

    p_corner_item = &(*p_schur_form_)(*cur_size - 1, *cur_size - 2);
    if (near_zero(*p_corner_item)) {
      *p_corner_item = 0;
      *cur_size -= 2;
    }
  }

  Vector3 find_matching_column(int cur_size) {
    Scalar trace = find_bottom_corner_trace(cur_size);
    Scalar det = find_bottom_corner_det(cur_size);
    BlockDynamic top_corner = p_schur_form_->topLeftCorner(3, 3);
    Matrix3 matching_block = top_corner * top_corner - trace * top_corner +
                             det * Matrix3::Identity();
    return matching_block.col(0);
  }

  Scalar find_bottom_corner_trace(int cur_size) {
    return p_schur_form_->block(cur_size - 1, cur_size - 1, 2, 2).trace();
  }

  Scalar find_bottom_corner_det(int cur_size) {
    return p_schur_form_->block(cur_size - 1, cur_size - 1, 2, 2).determinant();
  }

  bool near_zero(Scalar value) { return abs(value) < precision_; }

  Precision precision_;
  SchurForm* p_schur_form_;
  UnitaryMatrix* p_unitary_;
  Reflector reflector_;
  int data_size_;
};

};  // namespace schur_decomposition
