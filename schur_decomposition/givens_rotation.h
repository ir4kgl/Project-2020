#ifndef _SCHUR_DECOMPOSITION_GIVENS_ROTATION_H
#define _SCHUR_DECOMPOSITION_GIVENS_ROTATION_H

#include <cmath>
#include <type_traits>

#include "../eigen/Eigen/Dense"

namespace givens_rotation {

template <class Scalar>
struct AngleData {
  Scalar cos;
  Scalar sin;
};

template <class Scalar>
AngleData<Scalar> find_angle(Scalar x, Scalar y) {
  assert(x != 0 || y != 0);
  Scalar modulus = std::hypot(x, y);
  return {x / modulus, y / modulus};
}

template <typename Scalar>
class GivensRotator {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

 public:
  using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
  using DynamicMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using DynamicBlock = Eigen::Block<Eigen::Matrix<Scalar, -1, -1>>;

  GivensRotator() = default;
  GivensRotator(Scalar x, Scalar y) {
    AngleData<Scalar> angle = find_angle(x, y);
    rotary_matrix_ << angle.cos, -angle.sin, angle.sin, angle.cos;
  }

  void rotate_left(DynamicMatrix* out) const {
    assert(out);
    DynamicMatrix& old = *out;
    assert(old.rows() == 2);
    old = rotary_matrix_.transpose() * old;
  }

  void rotate_right(DynamicMatrix* out) const {
    assert(out);
    DynamicMatrix& old = *out;
    assert(old.cols() == 2);
    old *= rotary_matrix_;
  }

  void rotate_left(DynamicBlock old) const {
    assert(old.rows() == 2);
    old = rotary_matrix_.transpose() * old;
  }

  void rotate_right(DynamicBlock old) const {
    assert(old.cols() == 2);
    old *= rotary_matrix_;
  }

  Scalar cos() const { return rotary_matrix_(0, 0); }
  Scalar sin() const { return rotary_matrix_(1, 0); }

 private:
  Matrix2 rotary_matrix_;
};

};  // namespace givens_rotation

#endif
