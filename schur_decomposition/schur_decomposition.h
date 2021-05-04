
#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"
#include "../schur_decomposition/hessenberg_reduction.h"

namespace schur_decomposition {

template <typename Scalar>
class SchurDecomposition {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using givens_rotation::GivensRotator;
  using householder_reflection::HouseholderReflector;

  using SchurForm = Eigen::Matrix<Scalar, -1, -1>;
  using SquareMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using UnitaryMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using Precision = Scalar;

 public:
  SchurDecomposition(Precision precision) : precision_(precision) {
    assert(precision >= 0);
  }

  void run(const SquareMatrix& data, SchurForm* schur_form,
           UnitaryMatrix* unitary) {
    assert(data.rows() == data.cols());
    assert(schur_form);
    assert(unitary);

    data_size_ = data.rows();
    p_schur_form_ = schur_form;
    *p_schur_form_ = data;
    p_unitary_ = unitary;
    *p_unitary_ = UnitaryMatrix::Identity(data_size_, data_size_);
  }

  void set_precision(Precision precision) {
    assert(precision >= 0);
    precision_ = precision;
  }

  Precision get_precision() const { return precision_; }

 private:
  Precision precision_;
  SchurForm* p_schur_form_;
  UnitaryMatrix* p_unitary_;
  int data_size_;
};

};  // namespace schur_decomposition
