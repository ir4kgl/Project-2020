#ifndef _SCHUR_DECOMPOSITION_SCHUR_DECOMPOSITION_H
#define _SCHUR_DECOMPOSITION_SCHUR_DECOMPOSITION_H

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"
#include "../schur_decomposition/hessenberg_reduction.h"
#include "../schur_decomposition/householder_reflection.h"
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
    current_size_ = size() - 1;
    while (current_size_ >= 1) {
      take_QR_implicit_step();
      try_to_deflate();
    }

    extract_eigenvalues(eigenvalues);
  }

  void take_QR_implicit_step() {
    set_matching_column();
    restore_tridiagonal_form();
  }

  void set_matching_column() {
    Rotator rotator = Rotator(
        diagonals_.get_major_diagonal()(0) - choose_shift_approximation(),
        diagonals_.get_side_diagonal()(0));
    update_diagonals(rotator, 0);
    update_unitary(rotator, 0);
    if (current_size_ > 1) {
      save_current_bulge(rotator, 0);
      update_side_diagonal(rotator, 0);
    };
  }

  void restore_tridiagonal_form() {
    for (int step = 1; step <= current_size_ - 1; ++step) {
      Rotator rotator = Rotator(get_rotated_item(step), current_bulge_);
      update_diagonals(rotator, step);
      update_unitary(rotator, step);
      if (step != current_size_ - 1) {
        save_current_bulge(rotator, step);
        update_side_diagonal(rotator, step);
      }
    }
  }

  Scalar choose_shift_approximation() {
    if (current_size_ > 1) {
      return find_wilkinson_shift();
    }
    return find_eigenvalue();
  }

  Scalar find_eigenvalue() {
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

  void update_diagonals(const Rotator& rotator, int step) {
    DynamicMatrix square(2, 2);

    square << diagonals_.get_major_diagonal()(step),
        diagonals_.get_side_diagonal()(step),
        diagonals_.get_side_diagonal()(step),
        diagonals_.get_major_diagonal()(step + 1);

    rotator.rotate_left(&square);
    rotator.rotate_right(&square);

    diagonals_.get_major_diagonal()(step) = square(0, 0);
    diagonals_.get_major_diagonal()(step + 1) = square(1, 1);
    diagonals_.get_side_diagonal()(step) = square(0, 1);

    if (step > 0) {
      diagonals_.get_side_diagonal()(step - 1) *= rotator.cos();
      diagonals_.get_side_diagonal()(step - 1) +=
          rotator.sin() * current_bulge_;
    }
  }

  void update_side_diagonal(const Rotator& rotator, int step) {
    diagonals_.get_side_diagonal()(step + 1) *= rotator.cos();
  }

  void update_unitary(const Rotator& rotator, int step) {
    rotator.rotate_right(p_unitary_->block(0, step, size(), 2));
  }

  void save_current_bulge(const Rotator& rotator, int step) {
    current_bulge_ = diagonals_.get_side_diagonal()(step + 1) * rotator.sin();
  }

  Scalar get_rotated_item(int step) {
    return diagonals_.get_side_diagonal()(step - 1);
  }

  void try_to_deflate() {
    if (zero_under_diagonal()) {
      --current_size_;
    }
  }

  bool zero_under_diagonal() {
    return std::abs(diagonals_.get_side_diagonal()(current_size_ - 1)) <
           precision_ *
               (std::abs(diagonals_.get_major_diagonal()(current_size_ - 1)) +
                std::abs(diagonals_.get_major_diagonal()(current_size_)));
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

  bool near_zero(Scalar value) { return std::abs(value) < precision_; }

  int size() { return diagonals_.get_size(); }

  Precision precision_;
  TridiagonalSymmetric diagonals_;
  DynamicMatrix* p_unitary_;

  int current_size_;
  Scalar current_bulge_;
};

};  // namespace schur_decomposition

#endif
