#include <cmath>
#include <type_traits>

#include "Eigen/Dense"

namespace Math_tools {

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
class Givens_rotator {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
  using MatrixDynamic = Eigen::Matrix<Scalar, -1, -1>;
  using BlockDynamic = Eigen::Block<Eigen::Matrix<Scalar, -1, -1>>;

 public:
  Givens_rotator() = default;
  Givens_rotator(Scalar x, Scalar y) {
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

template <typename Scalar>
class Householder_reflector {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using VectorDynamic = Eigen::Matrix<Scalar, -1, 1>;
  using BlockDynamic = Eigen::Block<Eigen::Matrix<Scalar, -1, -1>>;

 public:
  Householder_reflector() = default;

  Householder_reflector(VectorDynamic vector) : direction_(std::move(vector)) {
    int size = direction_.rows();
    assert(size >= 2);
    if (direction_(0) > 0) {
      direction_(0) += direction_.norm();
    } else {
      direction_(0) -= direction_.norm();
    }
    direction_.normalize();
  }

  const VectorDynamic& direction() const { return direction_; }

  void reflect_left(BlockDynamic block) const {
    assert(block.rows() == direction_.rows());
    block -= direction_ * (2 * (direction_.transpose() * block));
  }

  void reflect_right(BlockDynamic block) const {
    assert(block.cols() == direction_.rows());
    block -= 2 * (block * direction_) * direction_.transpose();
  }

 private:
  VectorDynamic direction_;
};

};  // namespace Math_tools
