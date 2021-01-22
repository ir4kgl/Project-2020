#include <algorithm>
#include <cmath>

#include "Givens_rotation.h"
#include "Hessenberg_reduction.h"

namespace General_QR {

using namespace Householder_reflection;
using namespace General_reduction;

using namespace Eigen;

template <typename Scalar>
class Schur_decomposition {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using Reflector = Householder_reflector<Scalar>;
  using Hessenberg = Hessenberg_form<Scalar>;

  using MatrixDynamic = Matrix<Scalar, -1, -1>;
  using BlockDynamic = Block<Matrix<Scalar, -1, -1>>;
  using Vector2 = Matrix<Scalar, 2, 1>;
  using Vector3 = Matrix<Scalar, 3, 1>;
  using Matrix3 = Matrix<Scalar, 3, 3>;

 public:
  Schur_decomposition(Scalar precision, const MatrixDynamic& input,
                      MatrixDynamic* schur_vectors = NULL) {
    assert(precision >= 0.);
    assert(input.rows() == input.cols());

    init(input, schur_vectors);

    run_algorithm(precision, schur_vectors);
  }

  MatrixDynamic* get_triangular() { return &triangular_; }

 private:
  MatrixDynamic triangular_;

  void init(const MatrixDynamic& input, MatrixDynamic* schur_vectors) {
    Hessenberg reduced(input, schur_vectors);
    triangular_ = *reduced.get();
  }

  Matrix3 find_matching(int index) {
    Scalar trace = triangular_.block(index - 1, index - 1, 2, 2).trace();
    Scalar det = triangular_.block(index - 1, index - 1, 2, 2).determinant();

    BlockDynamic corner = triangular_.topLeftCorner(3, 3);

    Matrix3 temp = corner * corner - trace * corner + det * Matrix3::Identity();
    return temp;
  }

  void process_column(int corner, int step, const Vector3& column,
                      MatrixDynamic* schur_vectors) {
    int full_size = triangular_.rows();
    Reflector reflector(column);

    reflector.transform_left(triangular_(
        seq(step + 1, step + 3), seq(std::max(0, step), full_size - 1)));

    reflector.transform_right(triangular_(seq(0, std::min(corner, step + 4)),
                                          seq(step + 1, step + 3)));

    if (schur_vectors) {
      reflector.transform_right(
          (*schur_vectors)(seq(0, full_size - 1), seq(step + 1, step + 3)));
    }
  }

  void slide_bulge(int corner, int step, Vector3* out) {
    assert(out);
    Vector3& bulge_column = *out;

    int next_col = step + 1;

    bulge_column(0) = triangular_(next_col + 1, next_col);
    bulge_column(1) = triangular_(next_col + 2, next_col);

    if (next_col + 3 <= corner) {
      bulge_column(2) = triangular_(next_col + 3, next_col);
    }
  }

  void process_tail(int corner, MatrixDynamic* schur_vectors) {
    int full_size = triangular_.rows();
    int last_col = corner - 2;

    Vector2 bulge_column =
        triangular_(seq(last_col + 1, last_col + 2), seq(last_col, last_col));

    Reflector reflector(bulge_column);

    reflector.transform_left(
        triangular_(seq(corner - 1, corner), seq(corner - 2, full_size - 1)));

    reflector.transform_right(
        triangular_(seq(0, corner), seq(corner - 1, corner)));

    if (schur_vectors) {
      reflector.transform_right(
          (*schur_vectors)(seq(0, full_size - 1), seq(corner - 1, corner)));
    }
  }

  void remove_bulge(int corner, MatrixDynamic* schur_vectors) {
    if (triangular_.rows() > 3) {
      Vector3 bulge_colummn = triangular_(seq(1, 3), seq(0, 0));

      for (int step = 0; step <= corner - 3; ++step) {
        process_column(corner, step, bulge_colummn, schur_vectors);
        slide_bulge(corner, step, &bulge_colummn);
      }
    }

    process_tail(corner, schur_vectors);
  }

  void implicit_step(int corner, MatrixDynamic* schur_vectors) {
    Vector3 matching_column = find_matching(corner).col(0);

    int step = -1;
    process_column(corner, step, matching_column, schur_vectors);
    remove_bulge(corner, schur_vectors);
  }

  void deflate(Scalar precision, int* out) {
    assert(out);
    int& corner = *out;

    int first = corner - 1;
    int last = corner;

    Scalar& first_deflated_item = triangular_(last, first);

    if (std::abs(first_deflated_item) < precision) {
      first_deflated_item = 0.;
      --corner;
      return;
    }

    --first;
    --last;

    Scalar& next_deflated_item = triangular_(last, first);

    if (std::abs(next_deflated_item) < precision) {
      next_deflated_item = 0.;
      corner -= 2;
    }
  }

  void run_algorithm(Scalar precision, MatrixDynamic* schur_vectors) {
    int full_size = triangular_.rows();
    int actual_corner = full_size - 1;

    while (actual_corner >= 2) {
      implicit_step(actual_corner, schur_vectors);
      deflate(precision, &actual_corner);
    }
  }
};

};  // namespace General_QR

namespace Symmetric_QR {

using namespace Eigen;
using namespace Symmetric_reduction;
using namespace Givens_rotation;

template <typename Scalar>
class Schur_decomposition {
  static_assert(std::is_arithmetic_v<Scalar>,
                "Scalar must be arithmetic type!");

  using Tridiagonal = Tridiagonal_form<Scalar>;
  using Rotator = Givens_rotator<Scalar>;

  using MatrixDynamic = Matrix<Scalar, -1, -1>;
  using VectorDynamic = Matrix<Scalar, -1, 1>;

 public:
  Schur_decomposition(Scalar precision, const MatrixDynamic& input,
                      MatrixDynamic* eigenvectors = NULL) {
    assert(precision >= 0.);
    assert(input.rows() == input.cols());

    Tridiagonal diagonals(input, eigenvectors);
    run_algorithm(precision, &diagonals, eigenvectors);
  }

  VectorDynamic* get_eigenvalues() { return &eigenvalues_; }

 private:
  VectorDynamic eigenvalues_;

  MatrixDynamic get_square(int column, const Tridiagonal& diagonals) {
    int first = column;
    int last = column + 1;

    MatrixDynamic temp(2, 2);
    temp << diagonals.major(first), diagonals.side(last), diagonals.side(last),
        diagonals.major(last);

    return temp;
  }

  void put_square(const MatrixDynamic& square, int column, Tridiagonal* out) {
    assert(out);
    Tridiagonal& diagonals = *out;

    assert(square.rows() == 2);
    assert(square.cols() == 2);
    assert(std::abs(square(0, 1) - square(1, 0)) < 1e-6);

    int first = column;
    int last = column + 1;

    *diagonals.major(first) = square(0, 0);
    *diagonals.major(last) = square(1, 1);
    *diagonals.side(last) = square(0, 1);
  }

  void update_vectors(int column, const Rotator& rotator, MatrixDynamic* out) {
    if (out) {
      MatrixDynamic& vectors = *out;
      rotator.transform_right(vectors(all, seq(column, column + 1)));
    }
  }

  void process_column(int column, const Rotator& rotator, Tridiagonal* out,
                      MatrixDynamic* eigenvectors) {
    assert(out);
    Tridiagonal& diagonals = *out;

    MatrixDynamic square = get_square(column, diagonals);

    rotator.transform_left(&square);
    rotator.transform_right(&square);

    put_square(square, column, &diagonals);

    update_vectors(column, rotator, eigenvectors);
  }

  void remove_bulge(int corner, Scalar bulge, Tridiagonal* out,
                    MatrixDynamic* eigenvectors) {
    assert(out);
    Tridiagonal& diagonals = *out;

    for (int column = 1; column <= corner - 1; ++column) {
      Scalar kept = *diagonals.side(column);
      Scalar zeroed = bulge;

      Rotator rotator(kept, zeroed);

      *diagonals.side(column) = rotator.cos() * kept - rotator.sin() * zeroed;
      process_column(column, rotator, &diagonals, eigenvectors);

      if (column != corner - 1) {
        bulge = -*diagonals.side(column + 2) * rotator.sin();
        *diagonals.side(column + 2) *= rotator.cos();
      }
    }
  }

  Scalar wilkinson_shift(Scalar precision, int corner,
                         const Tridiagonal& diagonals) {
    int first = corner - 1;
    int last = corner;

    Scalar delta = (diagonals.major(first) - diagonals.major(last)) / 2.;
    Scalar hypot = std::hypot(delta, diagonals.side(last));

    if (delta > precision) {
      return diagonals.major(last) -
             std::pow(diagonals.side(last), 2) / (delta + hypot);
    }

    if (delta < -precision) {
      return diagonals.major(last) -
             std::pow(diagonals.side(last), 2) / (delta - hypot);
    }

    return diagonals.major(last) - std::abs(diagonals.side(last));
  }

  void implicit_step(Scalar precision, int corner, Tridiagonal* out,
                     MatrixDynamic* eigenvectors) {
    assert(out);
    Tridiagonal& diagonals = *out;

    Scalar shift = wilkinson_shift(precision, corner, diagonals);

    Scalar kept = *diagonals.major(0) - shift;
    Scalar zeroed = *diagonals.side(1);
    Rotator rotator(kept, zeroed);

    process_column(0, rotator, &diagonals, eigenvectors);

    Scalar bulge = -*diagonals.side(2) * rotator.sin();
    *diagonals.side(2) *= rotator.cos();

    remove_bulge(corner, bulge, &diagonals, eigenvectors);
  }

  void process_smallest_corner(Tridiagonal* out, MatrixDynamic* eigenvectors) {
    assert(out);
    Tridiagonal& diagonals = *out;

    Scalar kept = *diagonals.major(0);
    Scalar zeroed = *diagonals.side(1);

    Rotator rotator(kept, zeroed);

    int column = 0;
    process_column(column, rotator, &diagonals, eigenvectors);
  }

  void deflate(Scalar precision, const Tridiagonal& diagonals, int* out) {
    assert(out);
    int& corner = *out;

    Scalar deflated_item = diagonals.side(corner);

    if (std::abs(deflated_item) < precision) {
      --corner;
    }
  }

  void run_algorithm(Scalar precision, Tridiagonal* out,
                     MatrixDynamic* eigenvectors) {
    assert(out);
    Tridiagonal& diagonals = *out;

    int full_size = diagonals.major_diagonal()->rows();
    int actual_corner = full_size - 1;

    while (actual_corner > 0) {
      if (actual_corner != 1) {
        implicit_step(precision, actual_corner, &diagonals, eigenvectors);
      } else {
        process_smallest_corner(&diagonals, eigenvectors);
      }

      deflate(precision, diagonals, &actual_corner);
    }

    eigenvalues_ = *diagonals.major_diagonal();
  }
};

};  // namespace Symmetric_QR
