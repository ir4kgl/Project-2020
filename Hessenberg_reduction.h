#include "Householder_reflection.h"

namespace General_reduction {

using namespace Eigen;

using namespace Householder_reflection;

template <typename Scalar>
class Hessenberg_form {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using Reflector = Householder_reflector<Scalar>;

  using MatrixDynamic = Matrix<Scalar, -1, -1>;
  using VectorDynamic = Matrix<Scalar, -1, 1>;

 public:
  Hessenberg_form(const MatrixDynamic& input,
                  MatrixDynamic* transition = NULL) {
    assert(input.rows() == input.cols());
    int size = input.rows();

    transformed_ = input;

    if (transition) {
      assign_identity(size, transition);
    }

    reduce_to_hessenberg(transition);
  }

  MatrixDynamic* get() { return &transformed_; }

 private:
  MatrixDynamic transformed_;

  void assign_identity(int size, MatrixDynamic* out) {
    assert(out);
    MatrixDynamic& assigned = *out;

    assigned = MatrixDynamic::Identity(size, size);
  }

  void update_transition(int full_size, int step, const Reflector& reflector,
                         MatrixDynamic* out) {
    if (!out) return;
    MatrixDynamic& transition = *out;

    reflector.transform_right(
        transition(seq(0, full_size - 1), seq(step + 1, full_size - 1)));
  }

  void reduction_step(int step, const Reflector& reflector) {
    int full_size = transformed_.rows();

    reflector.transform_left(
        transformed_(seq(step + 1, full_size - 1), seq(step, full_size - 1)));

    reflector.transform_right(
        transformed_(seq(0, full_size - 1), seq(step + 1, full_size - 1)));
  }

  void reduce_to_hessenberg(MatrixDynamic* transition) {
    int full_size = transformed_.rows();

    for (int step = 0; step < full_size - 2; ++step) {
      Reflector reflector(
          transformed_(seq(step + 1, full_size - 1), seq(step, step)));

      reduction_step(step, reflector);

      update_transition(full_size, step, reflector, transition);
    }
  }
};

};  // namespace General_reduction

namespace Symmetric_reduction {

using namespace Householder_reflection;

template <typename Scalar>
class Tridiagonal_form {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using Reflector = Householder_reflector<Scalar>;

  using MatrixDynamic = Matrix<Scalar, -1, -1>;
  using VectorDynamic = Matrix<Scalar, -1, 1>;

 public:
  Tridiagonal_form(const MatrixDynamic& input,
                   MatrixDynamic* transition = NULL) {
    assert(input.rows() == input.cols());
    int size = input.rows();

    if (transition) {
      assign_identity(size, transition);
    }

    MatrixDynamic symmetric = input;
    reduce_to_tridiagonal(&symmetric, transition);
  }

  VectorDynamic* major_diagonal() { return &major_; }
  VectorDynamic* side_diagonal() { return &side_; }
  const VectorDynamic& major_diagonal() const { return major_; }
  const VectorDynamic& side_diagonal() const { return side_; }

  Scalar* major(int index) { return &major_(index); }
  Scalar* side(int index) { return &side_(index); }
  const Scalar& major(int index) const { return major_(index); }
  const Scalar& side(int index) const { return side_(index); }

 private:
  VectorDynamic major_;
  VectorDynamic side_;

  void assign_identity(int size, MatrixDynamic* out) {
    assert(out);
    MatrixDynamic& assigned = *out;

    assigned = MatrixDynamic::Identity(size, size);
  }

  void assign_major(const MatrixDynamic& from) { major_ = from.diagonal(0); }

  void assign_side(const MatrixDynamic& from) {
    int full_size = from.rows();

    side_.resize(full_size);
    side_.tail(full_size - 1) = from.diagonal(1);
    side_(0) = 0;
  }

  void update_transition(int full_size, int step, const Reflector& reflector,
                         MatrixDynamic* out) {
    if (!out) return;
    MatrixDynamic& transition = *out;

    reflector.transform_right(
        transition(seq(0, full_size - 1), seq(step + 1, full_size - 1)));
  }

  void reduction_step(int step, const Reflector& reflector,
                      MatrixDynamic* out) {
    assert(out);
    MatrixDynamic& symmetric = *out;
    int full_size = symmetric.rows();

    int zeroed_elems = full_size - step - 1;

    VectorDynamic first = *reflector.get();
    VectorDynamic second =
        symmetric.bottomRightCorner(full_size, zeroed_elems) * first;
    second.tail(zeroed_elems) -=
        first * (first.transpose() * second.tail(zeroed_elems));
    second *= 2;

    MatrixDynamic temp = first * second.transpose();

    symmetric.bottomLeftCorner(zeroed_elems, full_size) -= temp;
    symmetric.topRightCorner(full_size, zeroed_elems) -= temp.transpose();
  }

  void reduce_to_tridiagonal(MatrixDynamic* out, MatrixDynamic* transition) {
    assert(out);
    MatrixDynamic& symmetric = *out;

    int full_size = symmetric.rows();

    for (int step = 0; step < full_size - 2; ++step) {
      int col_index = step;
      int rows_count = full_size - col_index - 1;
      Reflector reflector(symmetric.col(col_index).bottomRows(rows_count));

      reduction_step(step, reflector, &symmetric);
      update_transition(full_size, step, reflector, transition);
    }

    assign_major(symmetric);
    assign_side(symmetric);
  }
};

};  // namespace Symmetric_reduction
