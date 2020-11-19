#include <Eigen/Dense>
#include <cmath>

namespace Givens_rotation {

template <typename Scalar>
class Givens_rotation {
 public:
  Givens_rotation(Scalar upper_val, Scalar lower_val);
  Givens_rotation() {}
  const Scalar& get_cos() const;
  const Scalar& get_sin() const;
  Eigen::Matrix<Scalar, 2, 2> left_rotation() const;
  Eigen::Matrix<Scalar, 2, 2> right_rotation() const;
  void rotate_tridiag(Scalar& diag_ff, Scalar& diag_ss, Scalar& off_diag) const;

 private:
  Scalar cos_giv;
  Scalar sin_giv;
};

template <typename Scalar>
Givens_rotation<Scalar>::Givens_rotation(Scalar upper_val, Scalar lower_val) {
  Scalar norm = std::hypot(std::abs(upper_val), std::abs(lower_val));
  cos_giv = upper_val / norm;
  sin_giv = -lower_val / norm;
}

template <typename Scalar>
const Scalar& Givens_rotation<Scalar>::get_cos() const {
  return cos_giv;
}

template <typename Scalar>
const Scalar& Givens_rotation<Scalar>::get_sin() const {
  return sin_giv;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 2> Givens_rotation<Scalar>::left_rotation() const {
  Eigen::Matrix<Scalar, 2, 2> lr;
  lr << cos_giv, -sin_giv, sin_giv, cos_giv;
  return lr;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 2, 2> Givens_rotation<Scalar>::right_rotation() const {
  Eigen::Matrix<Scalar, 2, 2> rr;
  rr << cos_giv, sin_giv, -sin_giv, cos_giv;
  return rr;
}

template <typename Scalar>
void Givens_rotation<Scalar>::rotate_tridiag(Scalar& diag_ff, Scalar& diag_ss,
                                             Scalar& off_diag) const {
  Scalar temp_00 = cos_giv * diag_ff - sin_giv * off_diag;
  Scalar temp_01 = cos_giv * off_diag - sin_giv * diag_ss;
  Scalar temp_10 = sin_giv * diag_ff + cos_giv * off_diag;
  Scalar temp_11 = sin_giv * off_diag + cos_giv * diag_ss;
  diag_ff = temp_00 * cos_giv - temp_01 * sin_giv;
  diag_ss = temp_10 * sin_giv + temp_11 * cos_giv;
  off_diag = temp_00 * sin_giv + temp_01 * cos_giv;
}

};  // namespace Givens_rotation
