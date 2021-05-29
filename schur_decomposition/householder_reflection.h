#ifndef _SCHUR_DECOMPOSITION_HOUSEHOLDER_REFLECTION_H
#define _SCHUR_DECOMPOSITION_HOUSEHOLDER_REFLECTION_H

#include <cmath>
#include <type_traits>

#include "../eigen/Eigen/Dense"

namespace householder_reflection {

template <typename Scalar>
class HouseholderReflector {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

 public:
  using DynamicMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using DynamicVector = Eigen::Matrix<Scalar, -1, 1>;
  using DynamicBlock = Eigen::Block<DynamicMatrix>;

  HouseholderReflector() = default;

  HouseholderReflector(DynamicVector vector) : direction_(std::move(vector)) {
    int size = direction_.rows();
    assert(size >= 2);
    if (direction_(0) > 0) {
      direction_(0) += direction_.norm();
    } else {
      direction_(0) -= direction_.norm();
    }
    direction_.normalize();
  }

  const DynamicVector& direction() const { return direction_; }

  void reflect_left(DynamicBlock block) const {
    assert(block.rows() == direction_.rows());
    block -= direction_ * (2 * (direction_.transpose() * block));
  }

  void reflect_right(DynamicBlock block) const {
    assert(block.cols() == direction_.rows());
    block -= 2 * (block * direction_) * direction_.transpose();
  }

 private:
  DynamicVector direction_;
};

};  // namespace householder_reflection

#endif
