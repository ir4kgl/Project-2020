#include <cmath>
#include <type_traits>

#include "Eigen/Dense"
#include "Math_tools.h"

namespace QR_algorithm {

using namespace Math_tools;

enum class ResultMode : unsigned char { Partial = 0, Comlete = 1 };

enum class MatrixMode : unsigned char { Symmetric = 0, Classic = 1 };

template <class Scalar>
struct Tridiagonal_symmetric {
  using MajorDiagonal = Eigen::Matrix<Scalar, -1, 1>;
  using SideDiagonal = Eigen::Matrix<Scalar, -1, 1>;
  MajorDiagonal major;
  SideDiagonal side;
};

template <typename Scalar>
class Algorithm {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using Precision = Scalar;
  using VectorDynamic = Eigen::Matrix<Scalar, -1, 1>;
  using MatrixDynamic = Eigen::Matrix<Scalar, -1, -1>;
  using BlockDynamic = Eigen::Block<Eigen::Matrix<Scalar, -1, -1>>;
  using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  using SquareMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using SymmetricMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using UnitaryMatrix = Eigen::Matrix<Scalar, -1, -1>;
  using SchurForm = Eigen::Matrix<Scalar, -1, -1>;
  using Eigenvalues = Eigen::Matrix<Scalar, -1, 1>;
  using TridiagonalSymmetric = Tridiagonal_symmetric<Scalar>;
  using Reflector = Householder_reflector<Scalar>;
  using Rotator = Givens_rotator<Scalar>;

 public:
  Algorithm(Precision precision) : precision_(precision) {
    assert(precision >= 0);
  }

  void run_classic(const SquareMatrix& data, SchurForm* out,
                   UnitaryMatrix* unitary) {
    assert(out);
    assert(unitary);
    p_schur_form_ = out;
    p_unitary_ = unitary;
    p_eigevalues_ = nullptr;
    p_diagonals_ = nullptr;
    run<ResultMode::Comlete, MatrixMode::Classic>(data);
  }

  void run_classic(const SquareMatrix& data, SchurForm* out) {
    assert(out);
    p_schur_form_ = out;
    p_unitary_ = nullptr;
    p_eigevalues_ = nullptr;
    p_diagonals_ = nullptr;
    run<ResultMode::Partial, MatrixMode::Classic>(data);
  }

  void run_symmetric(const SymmetricMatrix& data, Eigenvalues* eigenvalues,
                     UnitaryMatrix* unitary) {
    assert(eigenvalues);
    assert(unitary);
    p_eigevalues_ = eigenvalues;
    p_unitary_ = unitary;
    run<ResultMode::Comlete, MatrixMode::Symmetric>(data);
  }

  void run_symmetric(const SymmetricMatrix& data, Eigenvalues* eigenvalues) {
    assert(eigenvalues);
    p_eigevalues_ = eigenvalues;
    p_unitary_ = nullptr;
    run<ResultMode::Partial, MatrixMode::Symmetric>(data);
  }

  void set_precision(Precision precision) {
    assert(precision >= 0);
    precision_ = precision;
  }

  Precision get_precision() const { return precision_; }

 private:
  template <ResultMode RMode, MatrixMode MMode>
  void run(const SquareMatrix& data) {
    assert(data.rows() == data.cols());
    full_size_ = data.rows();
    SchurForm schur_form_temp;
    TridiagonalSymmetric diagonals_temp;
    if constexpr (MMode == MatrixMode::Symmetric) {
      p_schur_form_ = &schur_form_temp;
      p_diagonals_ = &diagonals_temp;
    }
    *p_schur_form_ = data;
    if constexpr (RMode == ResultMode::Comlete) {
      *p_unitary_ = UnitaryMatrix::Identity(full_size_, full_size_);
    }
    reduce_to_hessenberg_form<RMode, MMode>();
    actual_corner_ = full_size_ - 1;
    while (actual_corner_ >= 2) {
      if constexpr (MMode == MatrixMode::Classic) {
        implicit_step_classic<RMode>();
      } else {
        implicit_step_symmetric<RMode>();
      }
      deflate<MMode>();
    }
    if constexpr (MMode == MatrixMode::Symmetric) {
      process_smallest_corner<RMode>();
    }
  }

  template <ResultMode RMode, MatrixMode MMode>
  void reduce_to_hessenberg_form() {
    Reflector reflector;
    for (step_ = 0; step_ < full_size_ - 2; ++step_) {
      int col = step_;
      int rows = full_size_ - step_ - 1;
      reflector = Reflector(p_schur_form_->col(col).bottomRows(rows));
      if constexpr (MMode == MatrixMode::Classic) {
        reduce_column_classic(reflector);
      } else {
        reduce_column_symmetric(reflector);
      }
      if constexpr (RMode == ResultMode::Comlete) {
        reflector.reflect_right(
            p_unitary_->bottomRightCorner(full_size_, full_size_ - step_ - 1));
      }
    }
    if constexpr (MMode == MatrixMode::Symmetric) {
      extract_diagonals();
    }
  }

  void reduce_column_classic(const Reflector& reflector) {
    int zeroed_elems = full_size_ - step_ - 1;
    reflector.reflect_left(
        p_schur_form_->bottomRightCorner(zeroed_elems, full_size_ - step_));
    reflector.reflect_right(
        p_schur_form_->bottomRightCorner(full_size_, zeroed_elems));
  }

  void reduce_column_symmetric(const Reflector& reflector) {
    int zeroed_elems = full_size_ - step_ - 1;
    VectorDynamic first = reflector.direction();
    VectorDynamic second =
        p_schur_form_->bottomRightCorner(full_size_, zeroed_elems) * first;
    second.tail(zeroed_elems) -=
        first * (first.transpose() * second.tail(zeroed_elems));
    second *= 2;
    MatrixDynamic tmp = first * second.transpose();
    p_schur_form_->bottomLeftCorner(zeroed_elems, full_size_) -= tmp;
    p_schur_form_->topRightCorner(full_size_, zeroed_elems) -= tmp.transpose();
  }

  template <ResultMode RMode>
  void implicit_step_classic() {
    init<RMode>();
    int width = 3;
    Reflector reflector;
    for (; step_ <= actual_corner_ - 3; ++step_) {
      reflector = Reflector(p_schur_form_->block(step_ + 1, step_, width, 1));
      single_step<RMode>(reflector, width);
    }
    width = 2;
    reflector = Reflector(p_schur_form_->block(step_ + 1, step_, width, 1));
    single_step<RMode>(reflector, width);
  }

  template <ResultMode RMode>
  void implicit_step_symmetric() {
    Scalar bulge;
    init<RMode>(&bulge);
    Rotator rotator;
    for (; step_ <= actual_corner_ - 1; ++step_) {
      Scalar x = p_diagonals_->side(step_);
      Scalar y = bulge;
      rotator = Rotator(x, y);
      single_step<RMode>(rotator);
      p_diagonals_->side(step_) = rotator.cos() * x + rotator.sin() * y;
      if (step_ != actual_corner_ - 1) {
        bulge = p_diagonals_->side(step_ + 2) * rotator.sin();
        p_diagonals_->side(step_ + 2) *= rotator.cos();
      }
    }
  }

  template <MatrixMode MMode>
  void deflate() {
    int first = actual_corner_ - 1;
    int last = actual_corner_;
    Scalar* deflated_item;
    if constexpr (MMode == MatrixMode::Classic) {
      deflated_item = &(*p_schur_form_)(last, first);
    } else {
      deflated_item = &p_diagonals_->side(actual_corner_);
    }
    if (std::abs(*deflated_item) < precision_) {
      *deflated_item = 0;
      --actual_corner_;
      return;
    }
    if constexpr (MMode == MatrixMode::Classic) {
      --first;
      --last;
      deflated_item = &(*p_schur_form_)(last, first);
      if (std::abs(*deflated_item) < precision_) {
        *deflated_item = 0;
        actual_corner_ -= 2;
      }
    }
  }

  template <ResultMode RMode>
  void init() {
    Reflector reflector;
    step_ = -1;
    int width = 3;
    reflector = Reflector(find_matching_column());
    single_step<RMode>(reflector, width);
    ++step_;
  }

  template <ResultMode RMode>
  void init(Scalar* bulge) {
    Rotator rotator;
    step_ = 0;
    Scalar shift = wilkinson_shift();
    rotator = Rotator(p_diagonals_->major(0) - shift, p_diagonals_->side(1));
    single_step<RMode>(rotator);
    *bulge = p_diagonals_->side(2) * rotator.sin();
    p_diagonals_->side(2) *= rotator.cos();
    ++step_;
  }

  Scalar wilkinson_shift() {
    int first = actual_corner_ - 1;
    int last = actual_corner_;
    Scalar delta = p_diagonals_->major(first) - p_diagonals_->major(last);
    Scalar hypot = std::hypot(delta / 2, p_diagonals_->side(last));
    if (delta > 0) {
      return p_diagonals_->major(last) -
             std::pow(p_diagonals_->side(last), 2) / (delta / 2 + hypot);
    }
    if (delta < 0) {
      return p_diagonals_->major(last) -
             std::pow(p_diagonals_->side(last), 2) / (delta / 2 - hypot);
    }
    return p_diagonals_->major(last) - std::abs(p_diagonals_->side(last));
  }

  template <ResultMode RMode>
  void single_step(const Reflector& reflector, int width) {
    if (step_ >= 0) {
      reflector.reflect_left(
          p_schur_form_->block(step_ + 1, step_, width, full_size_ - step_));
    } else {
      reflector.reflect_left(
          p_schur_form_->block(step_ + 1, 0, width, full_size_));
    }
    reflector.reflect_right(p_schur_form_->block(
        0, step_ + 1, std::min(actual_corner_, step_ + 4) + 1, width));
    if constexpr (RMode == ResultMode::Comlete) {
      reflector.reflect_right(
          p_unitary_->block(0, step_ + 1, full_size_, width));
    }
  }

  template <ResultMode RMode>
  void single_step(const Rotator& rotator) {
    int first = step_;
    int last = step_ + 1;
    MatrixDynamic square(2, 2);
    square << p_diagonals_->major(first), p_diagonals_->side(last),
        p_diagonals_->side(last), p_diagonals_->major(last);
    rotator.rotate_left(&square);
    rotator.rotate_right(&square);
    p_diagonals_->major(first) = square(0, 0);
    p_diagonals_->major(last) = square(1, 1);
    p_diagonals_->side(last) = square(0, 1);
    if constexpr (RMode == ResultMode::Comlete) {
      rotator.rotate_right(p_unitary_->block(0, step_, full_size_, 2));
    }
  }

  void extract_diagonals() {
    p_diagonals_->major = p_schur_form_->diagonal(0);
    p_diagonals_->side.resize(full_size_);
    p_diagonals_->side.tail(full_size_ - 1) = p_schur_form_->diagonal(1);
    p_diagonals_->side(0) = 0;
  }

  Vector3 find_matching_column() {
    int start = actual_corner_ - 1;
    Scalar trace = p_schur_form_->block(start, start, 2, 2).trace();
    Scalar det = p_schur_form_->block(start, start, 2, 2).determinant();
    BlockDynamic corner = p_schur_form_->topLeftCorner(3, 3);
    Matrix3 tmp = corner * corner - trace * corner + det * Matrix3::Identity();
    return tmp.col(0);
  }

  template <ResultMode RMode>
  void process_smallest_corner() {
    MatrixDynamic square(2, 2);
    square << p_diagonals_->major(0), p_diagonals_->side(1),
        p_diagonals_->side(1), p_diagonals_->major(1);
    Scalar trace = square.trace();
    Scalar det = square.determinant();
    Scalar eigenvalue = (trace + std::sqrt(std::pow(trace, 2) - 4 * det)) / 2;
    Rotator rotator(p_diagonals_->major(0) - eigenvalue, p_diagonals_->side(1));
    step_ = 0;
    single_step<RMode>(rotator);
    *p_eigevalues_ = p_diagonals_->major;
  }

  Precision precision_;
  SchurForm* p_schur_form_;
  UnitaryMatrix* p_unitary_;
  Eigenvalues* p_eigevalues_;
  TridiagonalSymmetric* p_diagonals_;
  int full_size_;
  int actual_corner_;
  int step_;
};

};  // namespace QR_algorithm
