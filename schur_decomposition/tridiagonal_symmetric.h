#ifndef _SCHUR_DECOMPOSITION_TRIDIAGONAL_SYMMETRIC_H
#define _SCHUR_DECOMPOSITION_TRIDIAGONAL_SYMMETRIC_H

#include "../eigen/Eigen/Dense"

namespace tridiagonal_symmetric {

template <class Scalar>
class TridiagonalSymmetric {
 public:
  using DynamicMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using DynamicVector = Eigen::Matrix<Scalar, -1, 1>;

  TridiagonalSymmetric() = default;

  TridiagonalSymmetric(int size)
      : major_diagonal_(DynamicVector(size)),
        side_diagonal_(DynamicVector(size - 1)) {
    assert(size >= 1);
  }

  DynamicVector& get_major_diagonal() { return major_diagonal_; }

  DynamicVector& get_side_diagonal() { return side_diagonal_; }

  void set_diagonals(DynamicVector new_major, DynamicVector new_side) {
    assert(new_major.rows() == new_side.rows() + 1);
    major_diagonal_ = std::move(new_major);
    side_diagonal_ = std::move(new_side);
  }

  static TridiagonalSymmetric extract_diagonals(const DynamicMatrix& data) {
    TridiagonalSymmetric tmp;
    tmp.set_diagonals(data.diagonal(0), data.diagonal(1));
    return tmp;
  }

  int get_size() const {
    assert(major_diagonal_.rows() == side_diagonal_.rows() + 1);
    return major_diagonal_.rows();
  }

 private:
  DynamicVector major_diagonal_;
  DynamicVector side_diagonal_;
};

}  // namespace tridiagonal_symmetric

#endif
