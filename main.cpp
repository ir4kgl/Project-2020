#include <complex>
#include <iostream>

#include "QR_algorithm.h"

using namespace Eigen;
using namespace Math_tools;
using namespace QR_algorithm;

using complexd = std::complex<double>;
using MatrixXc = Matrix<std::complex<double>, -1, -1>;
using Rotator = Givens_rotator<double>;
using Reflector = Householder_reflector<double>;
using QR = Algorithm<double>;

void check_single_eigenvalue(complexd value, int size, const MatrixXc& base,
                             double precision) {
  MatrixXc characteristic = base - value * MatrixXc::Identity(size, size);
  auto tmp = std::abs(characteristic.determinant());
  assert(std::abs(characteristic.determinant()) < precision);
}

void check_conjugate_eigenvalues(Block<MatrixXd> values, int size,
                                 const MatrixXd& base, double precision) {
  complexd trace = values.trace();
  complexd det = values.determinant();
  complexd value = (trace + std::sqrt(std::pow(trace, 2) - 4. * det)) / 2.;
  check_single_eigenvalue(value, size, base, precision);
  check_single_eigenvalue(std::conj(value), size, base, precision);
}

class Algorithm_tests_classic {
 public:
  void run_all() {
    run_random(1);
    run_random(2);
    run_random(3);
    run_random(4);
    run_random(6);
    run_random(9);
    run_random(10);
    std::cout << "Passed Schur decompositon tests, general case\n";
  }

 private:
  void run_random(int size) {
    MatrixXd base = MatrixXd::Random(size, size);
    MatrixXd schur_form(size, size);
    MatrixXd unitary(size, size);
    QR algorithm(algorithm_precision);
    algorithm.run_classic(base, &schur_form, &unitary);
    for (int i = 0; i < size;) {
      if (i == size - 1 || std::abs(schur_form(i + 1, i)) < result_precision) {
        check_single_eigenvalue(schur_form(i, i), size, base, result_precision);
        i += 1;
        continue;
      }
      check_conjugate_eigenvalues(schur_form.block(i, i, 2, 2), size, base,
                                  result_precision);
      i += 2;
    }
    MatrixXd result = unitary.transpose() * base * unitary;
    double delta = (schur_form - result).norm();
    assert(delta < result_precision);
  }

  static constexpr const double algorithm_precision = 1e-10;
  static constexpr const double result_precision = 1e-7;
};

class Algorithm_tests_symmetric {
 public:
  void run_all() {
    run_random(2);
    run_random(3);
    run_random(4);
    run_random(6);
    run_random(9);
    run_random(10);
    std::cout << "Passed Schur decompositon tests, symmetic case\n";
  }

 private:
  void run_random(int size) {
    MatrixXd base = MatrixXd::Random(size, size);
    base *= base.transpose();
    VectorXd eigenvalues(size);
    MatrixXd unitary(size, size);
    QR algorithm(algorithm_precision);
    algorithm.run_symmetric(base, &eigenvalues, &unitary);
    for (int i = 0; i < size; ++i) {
      check_single_eigenvalue(eigenvalues(i), size, base, result_precision);
    }
    MatrixXd result = unitary.transpose() * base * unitary;
    double delta = (result.diagonal(0) - eigenvalues).norm();
    assert(delta < result_precision);
  }

  static constexpr const double algorithm_precision = 1e-10;
  static constexpr const double result_precision = 1e-7;
};

class Algorithm_tests {
 public:
  void run_all() {
    algorithm_tests_classic.run_all();
    algorithm_tests_symmetric.run_all();
    std::cout << "Passed all QR algorithm tests\n\n";
  }

  Algorithm_tests_classic algorithm_tests_classic;
  Algorithm_tests_symmetric algorithm_tests_symmetric;
};

class Givens_rotation_tests {
 public:
  void run_all() {
    run_random(1, 2);
    run_random(2, 1);
    run_random(2, 2);
    std::cout << "Passed Givens rotation tests\n\n";
  }

  static constexpr const double precision = 1e-16;

 private:
  void run_random(int cols, int rows) {
    if (cols == 2) {
      run_right(cols, rows);
    }
    if (rows == 2) {
      run_left(cols, rows);
    }
  }

  void run_left(int cols, int rows) {
    MatrixXd base = MatrixXd::Random(rows, cols);
    Rotator result(base(rows - 2, cols - 1), base(rows - 1, cols - 1));
    result.rotate_left(&base);
    assert(abs(base(rows - 1, cols - 1)) < precision);
  }

  void run_right(int cols, int rows) {
    MatrixXd base = MatrixXd::Random(rows, cols);
    Rotator result(base(rows - 1, cols - 2), base(rows - 1, cols - 1));
    result.rotate_right(&base);
    assert(abs(base(rows - 1, cols - 1)) < precision);
  }
};

class Householder_reflector_tests {
 public:
  void run_all() {
    run_random(2);
    run_random(3);
    run_random(14);
    run_random(50);
    run_random(100);
    std::cout << "Passed Householder reflector tests\n\n";
  }

  static constexpr const double precision = 1e-14;

 private:
  void run_random(int size) {
    MatrixXd base = MatrixXd::Random(size, size);
    MatrixXd base_t = base.transpose();
    Reflector reflector(base.col(0));
    reflector.reflect_left(base.block(0, 0, size, size));
    reflector.reflect_right(base_t.block(0, 0, size, size));
    MatrixXd left_result = base.col(0).tail(size - 1);
    assert(left_result.norm() < precision);
    MatrixXd right_result = base_t.row(0).tail(size - 1);
    assert(right_result.norm() < precision);
  }
};

class Complete_tests {
 public:
  void run_all() {
    givens_rotation_tests.run_all();
    householder_reflector_tests.run_all();
    algorithm_tests.run_all();
    std::cout << "Passed all tests\n";
  }

  Givens_rotation_tests givens_rotation_tests;
  Householder_reflector_tests householder_reflector_tests;
  Algorithm_tests algorithm_tests;
};

int main() {
  Complete_tests tests;
  tests.run_all();
  return 0;
}
