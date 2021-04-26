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

  using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
  using MatrixDynamic = Eigen::Matrix<Scalar, -1, -1>;
  using BlockDynamic = Eigen::Block<Eigen::Matrix<Scalar, -1, -1>>;

 public:
  GivensRotator() = default;
  GivensRotator(Scalar x, Scalar y) {
    angle_ = find_angle(x, y);
    rotary_matrix_ << angle_.cos, -angle_.sin, angle_.sin, angle_.cos;
  }

  void rotate_left(MatrixDynamic* out) const {
    assert(out);
    MatrixDynamic& old = *out;
    assert(old.rows() == 2);
    old = rotary_matrix_.transpose() * old;
  }

  void rotate_right(MatrixDynamic* out) const {
    assert(out);
    MatrixDynamic& old = *out;
    assert(old.cols() == 2);
    old *= rotary_matrix_;
  }

  void rotate_left(BlockDynamic old) const {
    assert(old.rows() == 2);
    old = rotary_matrix_.transpose() * old;
  }

  void rotate_right(BlockDynamic old) const {
    assert(old.cols() == 2);
    old *= rotary_matrix_;
  }

  Scalar cos() const { return angle_.cos; }
  Scalar sin() const { return angle_.sin; }

 private:
  AngleData<Scalar> angle_;
  Matrix2 rotary_matrix_;
};

};  // namespace givens_rotation
