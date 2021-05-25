#include "../eigen/Eigen/Dense"

namespace tridiagonal_symmetric {

template <class Scalar>
class TridiagonalSymmetric {
 public:
  using DynamicVector = Eigen::Matrix<Scalar, -1, 1>;

  TridiagonalSymmetric(int size)
      : major_diagonal_(DynamicVector(size)),
        side_diagonal_(DynamicVector(size - 1)) {
    assert(size >= 1);
  }

  const DynamicVector& get_major_diagonal() const { return major_diagonal_; }

  const DynamicVector& get_side_diagonal() const { return side_diagonal_; }

  void set_diagonals(const DynamicVector& new_major,
                     const DynamicVector& new_side) {
    assert(new_major.rows() == new_side.rows() + 1);
    major_diagonal_ = new_major;
    side_diagonal_ = new_side;
  }

  size_t get_size() const { return major_diagonal_.rows(); }

 private:
  DynamicVector major_diagonal_;
  DynamicVector side_diagonal_;
};

}  // namespace tridiagonal_symmetric
