#include <iomanip>
#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"

using std::abs;
using std::cout;
using std::endl;
using std::srand;
using std::time;

using Eigen::MatrixXd;
using givens_rotation::Givens_rotator;

namespace test_givens_rotator {

static constexpr const double precision = 1e-15;
static constexpr const int tests_counter = 1000;
static constexpr const int matrix_size = 30;

bool simple_check_left() {
  MatrixXd data = MatrixXd::Random(2, matrix_size);
  MatrixXd old_data = data;
  Givens_rotator rotator(data(0, 0), data(1, 0));
  rotator.rotate_left(&data);
  if (abs(data(1, 0) > precision)) {
    cout << "test failed in Givens_rotator::rotate_left():\n\n";
    cout << "input: M = " << old_data << "\n";
    cout << "expected M(1, 0) = 0;\n";
    cout << "rotation result:" << data << "\n";
    return false;
  }
  return true;
}

bool simple_check_right() {
  MatrixXd data = MatrixXd::Random(matrix_size, 2);
  MatrixXd old_data = data;
  Givens_rotator rotator(data(0, 0), data(0, 1));
  rotator.rotate_right(&data);
  if (abs(data(0, 1) > precision)) {
    cout << "test failed in Givens_rotator::rotate_left():\n\n";
    cout << "input: M =\n" << old_data << "\n\n";
    cout << "expected M(0, 1) = 0;\n\n";
    cout << "rotation result:\n" << data << "\n";
    return false;
  }
  return true;
}

void run() {
  for (int j = 0; j < tests_counter; ++j) {
    if (!simple_check_left() || !simple_check_right()) return;
  }

  cout << "Passed all Givens_rotator tests\n";
}

}  // namespace test_givens_rotator
