#include <cmath>
#include <type_traits>

#include "../eigen/Eigen/Dense"

namespace householder_reflection {

template <typename Scalar>
class HouseholderReflector {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using VectorDynamic = Eigen::Matrix<Scalar, -1, 1>;
  using BlockDynamic = Eigen::Block<Eigen::Matrix<Scalar, -1, -1>>;

 public:
  HouseholderReflector() = default;

  HouseholderReflector(VectorDynamic vector) : direction_(std::move(vector)) {
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

};  // namespace householder_reflection
