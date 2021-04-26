#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/householder_reflection.h"

using std::cout;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using householder_reflection::HouseholderReflector;

namespace test_householder_reflector {

static constexpr const long double precision = 1e-14;
static constexpr const int tests_counter = 1000;
static constexpr const int matrix_size = 100;

bool simple_check_left() {
  MatrixXd data = MatrixXd::Random(matrix_size, matrix_size);
  MatrixXd old_data = data;
  HouseholderReflector reflector((VectorXd)data.col(0));
  reflector.reflect_left(data.block(0, 0, matrix_size, matrix_size));

  if (data.col(0).tail(matrix_size - 1).norm() > precision) {
    cout << "test failed in HouseholderReflector::reflect_left():\n\n";
    cout << "input: M = \n" << old_data << "\n\n";
    cout << "expected M(2, 0) ... M(N, 0) = 0;\n";
    cout << "reflection result: M = \n" << data << "\n";
    return false;
  }
  return true;
}

bool simple_check_right() {
  MatrixXd data = MatrixXd::Random(matrix_size, matrix_size);
  MatrixXd old_data = data;
  HouseholderReflector reflector((VectorXd)data.row(0).transpose());
  reflector.reflect_right(data.block(0, 0, matrix_size, matrix_size));

  if (data.row(0).tail(matrix_size - 1).norm() > precision) {
    cout << "test failed in HouseholderReflector::reflect_left():\n\n";
    cout << "input: M = \n" << old_data << "\n\n";
    cout << "expected M(0, 2) ... M(0, N) = 0;\n";
    cout << "reflection result: M = \n" << data << "\n";
    return false;
  }
  return true;
}

void run() {
  for (int j = 0; j < tests_counter; ++j) {
    if (!simple_check_left() || !simple_check_right()) return;
  }

  cout << "Passed all HouseholderReflector tests\n";
}

}  // namespace test_householder_reflector
