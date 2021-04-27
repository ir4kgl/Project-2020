#include <iomanip>
#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"

namespace test_givens_rotator {

using std::abs;
using std::cout;
using std::endl;
using std::srand;
using std::time;

using givens_rotation::GivensRotator;
using DynamicMatrix = GivensRotator<double>::DynamicMatrix;

static constexpr const double precision = 1e-15;
static constexpr const int number_of_tests = 100;
static constexpr const int matrix_size_max = 64;

void process_left_check_failed(const DynamicMatrix& data,
                               const DynamicMatrix& old_data) {
  cout << "test failed in GivensRotator::rotate_left():\n\n";
  cout << "input: M = \n" << old_data << "\n\n";
  cout << "expected M(1, 0) = 0;\n\n";
  cout << "rotation result:\n" << data << "\n";
}

void process_right_check_failed(const DynamicMatrix& data,
                                const DynamicMatrix& old_data) {
  cout << "test failed in GivensRotator::rotate_right():\n\n";
  cout << "input: M = \n" << old_data << "\n\n";
  cout << "expected M(0, 1) = 0;\n\n";
  cout << "rotation result:\n" << data << "\n";
}

bool simple_check_left(int size) {
  DynamicMatrix data = DynamicMatrix::Random(2, size);
  DynamicMatrix old_data = data;
  GivensRotator rotator(data(0, 0), data(1, 0));
  rotator.rotate_left(&data);

  if (abs(data(1, 0) > precision)) {
    process_left_check_failed(data, old_data);
    return false;
  }
  return true;
}

bool simple_check_right(int size) {
  DynamicMatrix data = DynamicMatrix::Random(size, 2);
  DynamicMatrix old_data = data;
  GivensRotator rotator(data(0, 0), data(0, 1));
  rotator.rotate_right(&data);

  if (abs(data(0, 1) > precision)) {
    process_right_check_failed(data, old_data);
    return false;
  }
  return true;
}

void run() {
  for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
    for (int size = 1; size <= matrix_size_max; size *= 2) {
      srand(test_id);
      if (!simple_check_left(size) || !simple_check_right(size)) return;
      if (!simple_check_left(size + 1) || !simple_check_right(size + 1)) return;
    }
  }

  cout << "Passed all GivensRotator tests\n";
}

}  // namespace test_givens_rotator
