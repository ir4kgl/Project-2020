#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"
#include "../schur_decomposition/hessenberg_reduction.h"
#include "../schur_decomposition/tridiagonal_symmetric.h"

namespace schur_decomposition {

template <typename Scalar>
class SchurDecomposition {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

 public:
  using Precision = Scalar;
  using TridiagonalSymmetric =
      tridiagonal_symmetric::TridiagonalSymmetric<Scalar>;
  using Rotator = givens_rotation::GivensRotator<Scalar>;
  using HessenbergReduction = hessenberg_reduction::HessenbergReduction<Scalar>;

  using DynamicMatrix = HessenbergReduction::DynamicMatrix;
  using DynamicVector = HessenbergReduction::DynamicVector;

  SchurDecomposition(Precision precision) : precision_(precision) {
    assert(precision >= 0);
  }

  void run(const DynamicMatrix& data, DynamicVector* eigenvalues,
           DynamicMatrix* unitary) {
    set_internal_resources(data, eigenvalues, unitary);
    reduce_to_hessenberg_form(data);
    run_QR_algorithm(eigenvalues);
    free_internal_resources();
  }

  void set_precision(Precision precision) {
    assert(precision >= 0);
    precision_ = precision;
  }

  Precision get_precision() const { return precision_; }

 private:
  void reduce_to_hessenberg_form(const DynamicMatrix& data) {
    HessenbergReduction reduction;
    DynamicMatrix hessenberg_form = data;
    reduction.run(&hessenberg_form, p_unitary_);
    diagonals_ = TridiagonalSymmetric::extract_diagonals(hessenberg_form);
  }

  void run_QR_algorithm(DynamicVector* eigenvalues) {
    for (current_size_ = size() - 1; current_size_ >= 2;) {
      take_QR_implicit_step();
      try_to_deflate();
    }

    process_submatrix2();
    extract_eigenvalues(eigenvalues);
  }

  void take_QR_implicit_step() {
    make_shift();
    process_shift();
    finish();
  }

  void make_shift() {
    Scalar shift = find_wilkinson_shift();
    Rotator rotator = Rotator(diagonals_.get_major_diagonal()(0) - shift,
                              diagonals_.get_side_diagonal()(0));
    rotate_submatrix2(rotator, 0);
    rotator.rotate_right(p_unitary_->block(0, 0, size(), 2));
    current_bulge_ = diagonals_.get_side_diagonal()(1) * rotator.sin();
    diagonals_.get_side_diagonal()(1) *= rotator.cos();
  }

  void process_shift() {
    for (int step = 1; step <= current_size_ - 2; ++step) {
      Scalar tmp = diagonals_.get_side_diagonal()(step - 1);
      Rotator rotator = Rotator(tmp, current_bulge_);

      rotate_submatrix2(rotator, step);
      rotator.rotate_right(p_unitary_->block(0, step, size(), 2));
      diagonals_.get_side_diagonal()(step - 1) =
          rotator.cos() * tmp + rotator.sin() * current_bulge_;
      current_bulge_ = diagonals_.get_side_diagonal()(step + 1) * rotator.sin();
      diagonals_.get_side_diagonal()(step + 1) *= rotator.cos();
    }
  }

  void finish() {
    Scalar tmp = diagonals_.get_side_diagonal()(current_size_ - 2);
    Rotator rotator = Rotator(tmp, current_bulge_);

    rotate_submatrix2(rotator, current_size_ - 1);
    rotator.rotate_right(p_unitary_->block(0, current_size_ - 1, size(), 2));
    diagonals_.get_side_diagonal()(current_size_ - 2) =
        rotator.cos() * tmp + rotator.sin() * current_bulge_;
  }

  void rotate_submatrix2(const Rotator& rotator, int index) {
    DynamicMatrix square(2, 2);

    square << diagonals_.get_major_diagonal()(index),
        diagonals_.get_side_diagonal()(index),
        diagonals_.get_side_diagonal()(index),
        diagonals_.get_major_diagonal()(index + 1);

    rotator.rotate_left(&square);
    rotator.rotate_right(&square);

    diagonals_.get_major_diagonal()(index) = square(0, 0);
    diagonals_.get_major_diagonal()(index + 1) = square(1, 1);
    diagonals_.get_side_diagonal()(index) = square(0, 1);
  }

  void process_submatrix2() {
    Scalar eigenvalue = find_eigenvalue2();
    Rotator rotator = Rotator(diagonals_.get_major_diagonal()(0) - eigenvalue,
                              diagonals_.get_side_diagonal()(0));
    rotate_submatrix2(rotator, 0);
    rotator.rotate_right(p_unitary_->block(0, 0, size(), 2));
  }

  Scalar find_eigenvalue2() {
    DynamicMatrix square(2, 2);
    square << diagonals_.get_major_diagonal()(0),
        diagonals_.get_side_diagonal()(0), diagonals_.get_side_diagonal()(0),
        diagonals_.get_major_diagonal()(1);
    Scalar trace = square.trace();
    Scalar det = square.determinant();
    return (trace + std::sqrt(trace * trace - 4 * det)) / 2;
  }

  Scalar find_wilkinson_shift() {
    Scalar delta = diagonals_.get_major_diagonal()(current_size_ - 1) -
                   diagonals_.get_major_diagonal()(current_size_);
    Scalar hypot = std::hypot(
        delta / 2, diagonals_.get_side_diagonal()(current_size_ - 1));

    if (!near_zero(delta) && delta > 0) {
      return diagonals_.get_major_diagonal()(current_size_) -
             std::pow(diagonals_.get_side_diagonal()(current_size_ - 1), 2) /
                 (delta / 2 + hypot);
    }
    if (!near_zero(delta) && delta < 0) {
      return diagonals_.get_major_diagonal()(current_size_) -
             std::pow(diagonals_.get_side_diagonal()(current_size_ - 1), 2) /
                 (delta / 2 - hypot);
    }
    return diagonals_.get_major_diagonal()(current_size_) -
           std::abs(diagonals_.get_side_diagonal()(current_size_ - 1));
  }

  void try_to_deflate() {
    if (zero_under_diagonal()) {
      --current_size_;
    }
  }

  void set_internal_resources(const DynamicMatrix& data,
                              DynamicVector* eigenvalues,
                              DynamicMatrix* unitary) {
    assert(data.rows() == data.cols());
    assert(eigenvalues);
    assert(unitary);
    p_unitary_ = unitary;
  }

  void free_internal_resources() { diagonals_ = {}; }

  void extract_eigenvalues(DynamicVector* eigenvalues) {
    *eigenvalues = diagonals_.get_major_diagonal();
  }

  bool zero_under_diagonal() {
    return std::abs(diagonals_.get_side_diagonal()(current_size_ - 1)) <
           precision_ *
               (std::abs(diagonals_.get_major_diagonal()(current_size_ - 1)) +
                std::abs(diagonals_.get_major_diagonal()(current_size_)));
  }

  bool near_zero(Scalar value) { return std::abs(value) < precision_; }

  int size() { return diagonals_.get_size(); }

  Precision precision_;
  TridiagonalSymmetric diagonals_;
  DynamicMatrix* p_unitary_;

  int current_size_;
  Scalar current_bulge_;
};

};  // namespace schur_decomposition
