
#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"
#include "../schur_decomposition/hessenberg_reduction.h"

namespace schur_decomposition {

template <typename Scalar>
class SchurDecomposition {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using givens_rotation::GivensRotator;
  using householder_reflection::HouseholderReflector;

  using SchurForm = Eigen::Matrix<Scalar, -1, -1>;
  using SquareMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using UnitaryMatrix = Eigen::Matrix<Scalar, -1, -1>;

  void run(const SquareMatrix& data, SchurForm* out, UnitaryMatrix* unitary) {}
};
};  // namespace schur_decomposition