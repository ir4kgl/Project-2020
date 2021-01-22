#include "Eigen/Dense"

namespace Householder_reflection {

using namespace Eigen;

template <typename Scalar>
class Householder_reflector {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using VectorDynamic = Matrix<Scalar, -1, 1>;
  using BlockDynamic = Block<Matrix<Scalar, -1, -1>>;

 public:
  Householder_reflector(const VectorDynamic& input) {
    int dimension = input.rows();
    assert(dimension >= 2);

    basic_ = input;

    (basic_(0, 0) > 0)
        ? basic_ += basic_.norm() * VectorDynamic::Identity(dimension, 1)
        : basic_ -= basic_.norm() * VectorDynamic::Identity(dimension, 1);

    basic_.normalize();
  }

  VectorDynamic const* get() const { return &basic_; }

  void transform_left(BlockDynamic input) const {
    assert(input.rows() == basic_.rows());
    input -= 2 * basic_ * (basic_.transpose() * input);
  }

  void transform_right(BlockDynamic input) const {
    assert(input.cols() == basic_.rows());
    input -= 2 * (input * basic_) * basic_.transpose();
  }

 private:
  VectorDynamic basic_;
};

};  // namespace Householder_reflection
