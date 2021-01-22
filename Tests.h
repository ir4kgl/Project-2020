#include <complex>
#include <iostream>

#include "QR_algorithm.h"

namespace Tests {

using namespace Eigen;

using MatrixXc = Matrix<std::complex<double>, -1, -1>;

using Rotator = Givens_rotation::Givens_rotator<double>;
using Reflector = Householder_reflection::Householder_reflector<double>;
using Hessenberg_form = General_reduction::Hessenberg_form<double>;
using Tridiagonal_form = Symmetric_reduction::Tridiagonal_form<double>;
using Schur_decomposition_g = General_QR::Schur_decomposition<double>;
using Schur_decomposition_s = Symmetric_QR::Schur_decomposition<double>;

using complexd = std::complex<double>;

const double eps = 1e-6;
const double precision = 1e-16;

class rotation_test {
 public:
  int rows;
  int cols;

  void run_left() {
    MatrixXd base = MatrixXd::Random(rows, cols);
    Rotator result(base(rows - 2, cols - 1), base(rows - 1, cols - 1));

    result.transform_left(&base);

    assert(std::abs(base(rows - 1, cols - 1)) < eps);
  }

  void run_right() {
    MatrixXd base = MatrixXd::Random(rows, cols);
    Rotator result(base(rows - 1, cols - 2), base(rows - 1, cols - 1));

    result.transform_right(&base);

    double zeroed = std::abs(base(rows - 1, cols - 1));
    assert(zeroed < eps);
  }

  void run() {
    if (cols == 2) {
      run_right();
    }

    if (rows == 2) {
      run_left();
    }
  }
};

class Givens_rotation_tests {
 public:
  rotation_test test_21 = {2, 1};
  rotation_test test_12 = {1, 2};
  rotation_test test_22 = {2, 2};

  void run_all() {
    test_21.run();
    test_12.run();
    test_22.run();

    std::cout << "Passed Givens rotation tests\n\n";
  }
};

class reflector_test {
 public:
  int size;

  void run() {
    MatrixXd base = MatrixXd::Random(size, size);
    MatrixXd base_t = base.transpose();

    Reflector reflector(base.col(0));

    reflector.transform_left(base.block(0, 0, size, size));
    reflector.transform_right(base_t.block(0, 0, size, size));

    MatrixXd left_result = base.col(0).tail(size - 1);
    assert(left_result.norm() < eps);

    MatrixXd right_result = base_t.row(0).tail(size - 1);
    assert(right_result.norm() < eps);
  }
};

class Householder_reflector_tests {
 public:
  reflector_test test_2 = {2};
  reflector_test test_3 = {3};
  reflector_test test_14 = {14};
  reflector_test test_100 = {100};

  void run_all() {
    test_2.run();
    test_3.run();
    test_14.run();
    test_100.run();

    std::cout << "Passed Householder reflector tests\n\n";
  }
};

class reduction_test_g {
 public:
  int size;

  void run() {
    MatrixXd base = MatrixXd::Random(size, size);
    MatrixXd transition(size, size);

    Hessenberg_form hessenberg_form(base, &transition);
    MatrixXd fresh_result = transition.transpose() * base * transition;

    for (int j = 0; j < size - 2; ++j) {
      for (int i = j + 2; i < size; ++i) {
        assert(std::abs((*hessenberg_form.get())(i, j)) < eps);
      }
    }

    double delta = (*hessenberg_form.get() - fresh_result).norm();
    assert(delta < eps);
  }
};

class General_reduction_tests {
 public:
  reduction_test_g test_1 = {1};
  reduction_test_g test_2 = {2};
  reduction_test_g test_3 = {3};
  reduction_test_g test_14 = {14};
  reduction_test_g test_100 = {100};

  void run_all() {
    test_1.run();
    test_2.run();
    test_3.run();
    test_14.run();
    test_100.run();

    std::cout << "Passed reduction to Hessenberg form tests, general case\n";
  }
};

class reduction_test_s {
 public:
  int size;

  void run() {
    MatrixXd base = MatrixXd::Random(size, size);
    base *= base.transpose();
    MatrixXd transition(size, size);

    Tridiagonal_form diagonals(base, &transition);
    MatrixXd fresh_result = transition.transpose() * base * transition;

    double delta;

    delta = (fresh_result.diagonal(0) - *diagonals.major_diagonal()).norm();
    assert(delta < eps);

    delta = (fresh_result.diagonal(1) -
             (*diagonals.side_diagonal()).bottomRows(size - 1))
                .norm();
    assert(delta < eps);

    delta = (fresh_result.diagonal(-1) -
             (*diagonals.side_diagonal()).bottomRows(size - 1))
                .norm();
    assert(delta < eps);
  }
};

class Symmetric_reduction_tests {
 public:
  reduction_test_s test_1 = {1};
  reduction_test_s test_2 = {2};
  reduction_test_s test_3 = {3};
  reduction_test_s test_14 = {10};
  reduction_test_s test_100 = {100};

  void run_all() {
    test_1.run();
    test_2.run();
    test_3.run();
    test_14.run();
    test_100.run();

    std::cout << "Passed reduction to Hessenberg form tests, symmetric "
                 "tridiagonal case\n";
  }
};

class Hessenberg_reduction_tests {
 public:
  void run_all() {
    General_reduction_tests general_tests;
    Symmetric_reduction_tests symmetric_tests;

    general_tests.run_all();
    symmetric_tests.run_all();

    std::cout << "Passed all reduction to Hessenberg form tests\n\n";
  }
};

void check_single_eigenvalue(complexd value, int size, const MatrixXd& base) {
  MatrixXc characteristic = base - value * MatrixXc::Identity(size, size);
  assert(std::abs((characteristic).determinant()) < eps);
}

void check_conjugate_eigenvalues(Block<MatrixXd> values, int size,
                                 const MatrixXd& base) {
  complexd trace = values.trace();
  complexd det = values.determinant();

  complexd value = (trace + std::sqrt(std::pow(trace, 2.) - 4. * det)) / 2.;

  check_single_eigenvalue(value, size, base);
  check_single_eigenvalue(std::conj(value), size, base);
}

class QR_test_s {
 public:
  int size;

  void run() {
    MatrixXd base = MatrixXd::Random(size, size);
    base *= base.transpose();

    MatrixXd eigenvectors(size, size);

    Schur_decomposition_s schur_decomposition(precision, base, &eigenvectors);

    VectorXd& eigenvalues = *schur_decomposition.get_eigenvalues();

    for (int i = 0; i < size; ++i) {
      check_single_eigenvalue(eigenvalues(i), size, base);
    }

    MatrixXd fresh_result = eigenvectors.transpose() * base * eigenvectors;
    double delta = (fresh_result.diagonal(0) - eigenvalues).norm();

    assert(delta < eps);
  }
};

class Symmetric_algorithm_tests {
 public:
  QR_test_s test1 = {1};
  QR_test_s test2 = {2};
  QR_test_s test3 = {3};
  QR_test_s test4 = {4};
  QR_test_s test7 = {7};
  QR_test_s test9 = {9};
  QR_test_s test10 = {10};

  void run_all() {
    test1.run();
    test2.run();
    test3.run();
    test4.run();
    test7.run();
    test9.run();
    test10.run();

    std::cout << "Passed Schur decompositon tests, symmetic case\n";
  }
};

class QR_test_g {
 public:
  int size;

  void run() {
    MatrixXd base = MatrixXd::Random(size, size);
    MatrixXd schur_vectors(size, size);

    Schur_decomposition_g schur_decomposition(precision, base, &schur_vectors);

    MatrixXd triangular = *schur_decomposition.get_triangular();

    for (int i = 0; i < size;) {
      if (i == size - 1 || std::abs(triangular(i + 1, i)) < eps) {
        check_single_eigenvalue(triangular(i, i), size, base);
        i += 1;
        continue;
      }

      check_conjugate_eigenvalues(triangular(seq(i, i + 1), seq(i, i + 1)),
                                  size, base);
      i += 2;
    }

    MatrixXd fresh_result = schur_vectors.transpose() * base * schur_vectors;

    double delta = (triangular - fresh_result).norm();
    assert(delta < eps);
  }
};

class General_algorithm_tests {
 public:
  QR_test_g test1 = {1};
  QR_test_g test2 = {2};
  QR_test_g test3 = {3};
  QR_test_g test4 = {4};
  QR_test_g test7 = {7};
  QR_test_g test9 = {9};
  QR_test_g test10 = {10};

  void run_all() {
    test1.run();
    test2.run();
    test3.run();
    test4.run();
    test7.run();
    test9.run();
    test10.run();

    std::cout << "Passed Schur decompositon tests, general case\n";
  }
};

class Algorithm_tests {
 public:
  General_algorithm_tests general_QR_tests;
  Symmetric_algorithm_tests symmetric_QR_tests;

  void run_all() {
    general_QR_tests.run_all();
    symmetric_QR_tests.run_all();

    std::cout << "Passed all QR algorithm tests\n\n";
  }
};

class Complete_tests {
 public:
  Givens_rotation_tests givens_rotation_tests;
  Householder_reflector_tests householder_reflector_tests;
  Hessenberg_reduction_tests hessenberg_reduction_tests;
  Algorithm_tests algorithm_tests;

  void run_all() {
    givens_rotation_tests.run_all();
    householder_reflector_tests.run_all();
    hessenberg_reduction_tests.run_all();
    algorithm_tests.run_all();

    std::cout << "Passed all tests\n";
  }
};

};  // namespace Tests
