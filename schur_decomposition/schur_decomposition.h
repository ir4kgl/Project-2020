#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"
#include "../schur_decomposition/hessenberg_reduction.h"

namespace schur_decomposition {

using std::abs;
using std::is_arithmetic_v;
using std::min;
using std::pow;
using std::sqrt;

template <typename Scalar>
class SchurDecomposition {
  static_assert(is_arithmetic_v<Scalar>, "Scalar must be arithmetic type!");

 public:
  using DynamicMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using DynamicVector = Eigen::Matrix<Scalar, -1, 1>;
  using Precision = Scalar;

  using Rotator = givens_rotation::GivensRotator<Scalar>;
  using HessenbergReduction = hessenberg_reduction::HessenbergReduction<Scalar>;
  using TridiagonalSymmetric =
      tridiagonal_symmetric::TridiagonalSymmetric<Scalar>;

  SchurDecomposition(Precision precision) : precision_(precision) {
    assert(precision >= 0);
  }

  void run(const DynamicMatrix& data, DynamicVector* eigenvalues,
           DynamicMatrix* unitary) {
    assert(data.rows() == data.cols());
    assert(eigenvalues);
    assert(unitary);

    data_size_ = data.rows();
    p_eigenvalues_ = eigenvalues;
    p_unitary_ = unitary;

    reduce_to_hessenberg_form(data);
    run_QR_algorithm();
  }

  void set_precision(Precision precision) {
    assert(precision >= 0);
    precision_ = precision;
  }

  Precision get_precision() const { return precision_; }

 private:
  void reduce_to_hessenberg_form(const DynamicMatrix& data) {
    HessenbergReduction reduction;
    reduction.run(data, &diagonals_, p_unitary_);
  }

  void run_QR_algorithm() {
    Scalar bulge;

    for (int cur_size = data_size_ - 1; cur_size >= 2;) {
      make_shift(cur_size, &bulge);
      process_shift(cur_size, &bulge);
      finish(cur_size, bulge);
      deflate(&cur_size);
    }

    process_submatrix2();
    *p_eigenvalues_ = diagonals_.get_major_diagonal();
    diagonals_ = {};
  }

  void make_shift(int cur_size, Scalar* bulge) {
    Scalar shift = wilkinson_shift(cur_size);
    rotator_ = Rotator(diagonals_.get_major_diagonal()(0) - shift,
                       diagonals_.get_side_diagonal()(0));
    rotate_twice(0);
    rotator_.rotate_right(p_unitary_->block(0, 0, data_size_, 2));
    *bulge = diagonals_.get_side_diagonal()(1) * rotator_.sin();
    diagonals_.get_side_diagonal()(1) *= rotator_.cos();
  }

  void process_shift(int cur_size, Scalar* bulge) {
    for (int step = 1; step <= cur_size - 2; ++step) {
      Scalar tmp = diagonals_.get_side_diagonal()(step - 1);
      rotator_ = Rotator(tmp, *bulge);

      rotate_twice(step);
      rotator_.rotate_right(p_unitary_->block(0, step, data_size_, 2));
      diagonals_.get_side_diagonal()(step - 1) =
          rotator_.cos() * tmp + rotator_.sin() * (*bulge);
      *bulge = diagonals_.get_side_diagonal()(step + 1) * rotator_.sin();
      diagonals_.get_side_diagonal()(step + 1) *= rotator_.cos();
    }
  }

  void finish(int cur_size, Scalar bulge) {
    Scalar tmp = diagonals_.get_side_diagonal()(cur_size - 2);
    rotator_ = Rotator(tmp, bulge);

    rotate_twice(cur_size - 1);
    rotator_.rotate_right(p_unitary_->block(0, cur_size - 1, data_size_, 2));
    diagonals_.get_side_diagonal()(cur_size - 2) =
        rotator_.cos() * tmp + rotator_.sin() * bulge;

    rotator_ = {};
  }

  void rotate_twice(int index) {
    DynamicMatrix square(2, 2);

    square << diagonals_.get_major_diagonal()(index),
        diagonals_.get_side_diagonal()(index),
        diagonals_.get_side_diagonal()(index),
        diagonals_.get_major_diagonal()(index + 1);

    rotator_.rotate_left(&square);
    rotator_.rotate_right(&square);

    diagonals_.get_major_diagonal()(index) = square(0, 0);
    diagonals_.get_major_diagonal()(index + 1) = square(1, 1);
    diagonals_.get_side_diagonal()(index) = square(0, 1);
  }

  void process_submatrix2() {
    Scalar eigenvalue = find_eigenvalue2();
    rotator_ = Rotator(diagonals_.get_major_diagonal()(0) - eigenvalue,
                       diagonals_.get_side_diagonal()(0));
    rotate_twice(0);
    rotator_.rotate_right(p_unitary_->block(0, 0, data_size_, 2));
  }

  Scalar find_eigenvalue2() {
    DynamicMatrix square(2, 2);
    square << diagonals_.get_major_diagonal()(0),
        diagonals_.get_side_diagonal()(0), diagonals_.get_side_diagonal()(0),
        diagonals_.get_major_diagonal()(1);
    Scalar trace = square.trace();
    Scalar det = square.determinant();
    return (trace + sqrt(trace * trace - 4 * det)) / 2;
  }

  Scalar wilkinson_shift(int cur_size) {
    Scalar delta = diagonals_.get_major_diagonal()(cur_size - 1) -
                   diagonals_.get_major_diagonal()(cur_size);
    Scalar hypot =
        std::hypot(delta / 2, diagonals_.get_side_diagonal()(cur_size - 1));

    if (!near_zero(delta) && delta > 0) {
      return diagonals_.get_major_diagonal()(cur_size) -
             pow(diagonals_.get_side_diagonal()(cur_size - 1), 2) /
                 (delta / 2 + hypot);
    }
    if (!near_zero(delta) && delta < 0) {
      return diagonals_.get_major_diagonal()(cur_size) -
             pow(diagonals_.get_side_diagonal()(cur_size - 1), 2) /
                 (delta / 2 - hypot);
    }
    return diagonals_.get_major_diagonal()(cur_size) -
           abs(diagonals_.get_side_diagonal()(cur_size - 1));
  }

  void deflate(int* p_cur_size) {
    if (check_convergence(*p_cur_size)) {
      --(*p_cur_size);
      return;
    }
  }

  bool check_convergence(int p_cur_size) {
    return abs(diagonals_.get_side_diagonal()(p_cur_size - 1)) <
           precision_ * (abs(diagonals_.get_major_diagonal()(p_cur_size - 1)) +
                         abs(diagonals_.get_major_diagonal()(p_cur_size)));
  }

  bool near_zero(Scalar value) { return abs(value) < precision_; }

  Precision precision_;
  DynamicVector* p_eigenvalues_;
  DynamicMatrix* p_unitary_;

  TridiagonalSymmetric diagonals_;
  Rotator rotator_;
  int data_size_;
};

};  // namespace schur_decomposition
