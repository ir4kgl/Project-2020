#include "../eigen/Eigen/Dense"

namespace tridiagonal_symmetric {

template <class Scalar>
class TridiagonalSymmetric {
 public:
  using DynamicVector = Eigen::Matrix<Scalar, -1, 1>;

  TridiagonalSymmetric() = default;

  TridiagonalSymmetric(int size) : size_(size) {
    assert(size >= 2);
    major_diagonal_ = DynamicVector(size_);
    side_diagonal_ = DynamicVector(size_ - 1);
  }

  DynamicVector& get_major_diagonal() { return major_diagonal_; }
  const DynamicVector& get_major_diagonal() const { return major_diagonal_; }
  void set_major_diagonal(const DynamicVector& new_major) {
    major_diagonal_ = new_major;
  }

  DynamicVector& get_side_diagonal() { return side_diagonal_; }
  const DynamicVector& get_side_diagonal() const { return side_diagonal_; }
  void set_side_diagonal(const DynamicVector& new_side) {
    major_diagonal_ = new_side;
  }

  size_t get_size() const { return size_; }

 private:
  DynamicVector major_diagonal_;
  DynamicVector side_diagonal_;
  size_t size_;
};

}  // namespace tridiagonal_symmetric
