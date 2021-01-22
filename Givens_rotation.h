#include <cmath>
#include <type_traits>

#include "Eigen/Dense"

namespace Givens_rotation {

using namespace Eigen;

template <typename Scalar>
Scalar find_cos(Scalar kept, Scalar zeroed) {
  assert(kept != 0. || zeroed != 0.);
  return kept / std::hypot(kept, zeroed);
}

template <typename Scalar>
Scalar find_sin(Scalar kept, Scalar zeroed) {
  assert(kept != 0. || zeroed != 0.);
  return -zeroed / std::hypot(kept, zeroed);
}

template <typename Scalar>
class Givens_rotator {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using Matrix2 = Matrix<Scalar, 2, 2>;
  using MatrixDynamic = Matrix<Scalar, -1, -1>;
  using BlockDynamic = Block<Matrix<Scalar, -1, -1>, -1, -1, true>;

 public:
  Givens_rotator(Scalar kept, Scalar zeroed) {
    Scalar cos = find_cos(kept, zeroed);
    Scalar sin = find_sin(kept, zeroed);

    rotary_matrix_ << cos, sin, -sin, cos;
  }

  void transform_left(MatrixDynamic* out) const {
    assert(out);
    MatrixDynamic& old = *out;

    assert(old.rows() == 2);
    old = rotary_matrix_.transpose() * old;
  }

  void transform_right(MatrixDynamic* out) const {
    assert(out);
    MatrixDynamic& old = *out;

    assert(old.cols() == 2);
    old *= rotary_matrix_;
  }

  void transform_left(BlockDynamic old) const {
    assert(old.rows() == 2);
    old = rotary_matrix_.transpose() * old;
  }

  void transform_right(BlockDynamic old) const {
    assert(old.cols() == 2);
    old *= rotary_matrix_;
  }

  const Scalar& cos() const { return rotary_matrix_(0, 0); }
  const Scalar& sin() const { return rotary_matrix_(0, 1); }

 private:
  Matrix2 rotary_matrix_;
};

};  // namespace Givens_rotation
