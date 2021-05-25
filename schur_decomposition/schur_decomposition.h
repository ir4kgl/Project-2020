#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"
#include "../schur_decomposition/hessenberg_reduction.h"

namespace schur_decomposition {

template <typename Scalar>
class SchurDecomposition {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

 public:
  using HessenbergReduction = hessenberg_reduction::HessenbergReduction<Scalar>;
  using Rotator = givens_rotation::GivensRotator<Scalar>;
  using Reflector = HessenbergReduction::HouseholderReflector;

  using DynamicMatrix = HessenbergReduction::DynamicMatrix;
  using DynamicBlock = Eigen::Block<DynamicMatrix>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using Precision = Scalar;

  SchurDecomposition(Precision precision) : precision_(precision) {
    assert(precision >= 0);
  }

  void run(const DynamicMatrix& data, DynamicMatrix* schur_form,
           DynamicMatrix* unitary) {
    set_internal_resources(data, schur_form, unitary);
    reduce_to_hessenberg_form();
    run_QR_algorithm();
  }

  void set_precision(Precision precision) {
    assert(precision >= 0);
    precision_ = precision;
  }

  Precision get_precision() const { return precision_; }

 private:
  void reduce_to_hessenberg_form() {
    HessenbergReduction reduction;
    reduction.run(p_schur_form_, p_unitary_);
  }

  void run_QR_algorithm() {
    cur_size_ = size() - 1;
    while (cur_size_ >= 2) {
      make_QR_iteration();
      try_deflate();
    }
  }

  void make_QR_iteration() {
    set_matching_column();
    restore_hessenberg_form();
  }

  void set_matching_column() {
    Reflector reflector = Reflector(find_matching_column());

    reflector.reflect_left(p_schur_form_->block(0, 0, 3, size()));
    reflector.reflect_right(
        p_schur_form_->block(0, 0, std::min(cur_size_, 3) + 1, 3));
    reflector.reflect_right(p_unitary_->block(0, 0, size(), 3));
  }

  void restore_hessenberg_form() {
    int step = 0;
    for (; step <= cur_size_ - 3; ++step) {
      Reflector reflector =
          Reflector(p_schur_form_->block(step + 1, step, 3, 1));
      update_schur_form(reflector, step, 3);
      update_unitary(reflector, step, 3);
    }

    Reflector reflector = Reflector(p_schur_form_->block(step + 1, step, 2, 1));
    update_schur_form(reflector, step, 2);
    update_unitary(reflector, step, 2);
  }

  void update_schur_form(const Reflector& reflector, int step, int length) {
    reflector.reflect_left(
        p_schur_form_->block(step + 1, step, length, size() - step));
    reflector.reflect_right(p_schur_form_->block(
        0, step + 1, std::min(cur_size_, step + 4) + 1, length));
  }

  void update_unitary(const Reflector& reflector, int step, int length) {
    reflector.reflect_right(p_unitary_->block(0, step + 1, size(), length));
  }

  void try_deflate() {
    if (near_zero((*p_schur_form_)(cur_size_, cur_size_ - 1))) {
      deflate_once();
      return;
    }

    if (near_zero((*p_schur_form_)(cur_size_ - 1, cur_size_ - 2))) {
      deflate_twice();
    }
  }

  void deflate_once() {
    (*p_schur_form_)(cur_size_, cur_size_ - 1) = 0;
    cur_size_ -= 1;
  }

  void deflate_twice() {
    (*p_schur_form_)(cur_size_ - 1, cur_size_ - 2) = 0;
    cur_size_ -= 2;
  }

  Vector3 find_matching_column() {
    Scalar trace = find_bottom_corner_trace();
    Scalar det = find_bottom_corner_det();

    DynamicBlock top_corner = p_schur_form_->topLeftCorner(3, 3);
    Matrix3 starter_block = top_corner * top_corner - trace * top_corner +
                            det * Matrix3::Identity();
    return starter_block.col(0);
  }

  Scalar find_bottom_corner_trace() {
    return p_schur_form_->block(cur_size_ - 1, cur_size_ - 1, 2, 2).trace();
  }

  Scalar find_bottom_corner_det() {
    return p_schur_form_->block(cur_size_ - 1, cur_size_ - 1, 2, 2)
        .determinant();
  }

  void set_internal_resources(const DynamicMatrix& data,
                              DynamicMatrix* schur_form,
                              DynamicMatrix* unitary) {
    assert(schur_form);
    p_schur_form_ = schur_form;
    assert(data.rows() == data.cols());
    *p_schur_form_ = data;

    assert(unitary);
    p_unitary_ = unitary;
  }

  bool near_zero(Scalar value) { return std::abs(value) < precision_; }

  int size() {
    assert(p_schur_form_->rows() == p_schur_form_->cols());
    return p_schur_form_->rows();
  }

  Precision precision_;
  DynamicMatrix* p_schur_form_;
  DynamicMatrix* p_unitary_;
  int cur_size_;
};

};  // namespace schur_decomposition
