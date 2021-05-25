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

  void set_diagonals(const DynamicVector& new_major,
                     const DynamicVector& new_side) {
    assert(new_major.rows() == new_side.rows() + 1);
    major_diagonal_ = new_major;
    side_diagonal_ = new_side;
  }

  static TridiagonalSymmetric extract_diagonals(const DynamicMatrix& data) {
    TridiagonalSymmetric tmp;
    tmp.set_diagonals(data.diagonal(0), data.diagonal(1));
    return tmp;
  }

  size_t get_size() const { return major_diagonal_.rows(); }

 private:
  DynamicVector major_diagonal_;
  DynamicVector side_diagonal_;
};

}  // namespace tridiagonal_symmetric
