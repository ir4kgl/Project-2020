#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/givens_rotation.h"

namespace test_givens_rotator {

using std::abs;
using std::cout;
using std::srand;

using givens_rotation::GivensRotator;
using DynamicMatrix = GivensRotator<double>::DynamicMatrix;

constexpr const long double result_precision = 1e-15;
constexpr const int number_of_tests = 200;
constexpr const int number_of_skipped_tests = 50;
constexpr const int matrix_size_max = 100;

void process_left_check_failed(const DynamicMatrix& data,
                               const DynamicMatrix& old_data, int test_id) {
  cout << "test failed in GivensRotator::rotate_left():\n\n";
  cout << "input: M = \n" << old_data << "\n\n";
  cout << "expected M(1, 0) = 0;\n\n";
  cout << "rotation result:\n" << data << "\n";
  cout << "test id:\t" << test_id << "\n";
}

void process_right_check_failed(const DynamicMatrix& data,
                                const DynamicMatrix& old_data, int test_id) {
  cout << "test failed in GivensRotator::rotate_right():\n\n";
  cout << "input: M = \n" << old_data << "\n\n";
  cout << "expected M(0, 1) = 0;\n\n";
  cout << "rotation result:\n" << data << "\n";
  cout << "test id:\t" << test_id << "\n";
}

bool simple_check_left(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(2, size);
  DynamicMatrix old_data = data;
  GivensRotator rotator(data(0, 0), data(1, 0));
  rotator.rotate_left(&data);

  if (abs(data(1, 0)) > result_precision) {
    process_left_check_failed(data, old_data, test_id);
    return false;
  }
  return true;
}

bool simple_check_right(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(size, 2);
  DynamicMatrix old_data = data;
  GivensRotator rotator(data(0, 0), data(0, 1));
  rotator.rotate_right(&data);

  if (abs(data(0, 1)) > result_precision) {
    process_right_check_failed(data, old_data, test_id);
    return false;
  }
  return true;
}

void run_stress_testing() {
  for (int size = 1; size <= matrix_size_max; ++size) {
    for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
      srand(test_id);
      if (!simple_check_left(size, test_id)) return;
      if (!simple_check_right(size, test_id)) return;
    }
  }
  cout << "Passed GivensRotator stress testing\n";
  cout << "Number of tests: " << number_of_tests << "\n";
  cout << "Maximum matrix size: " << matrix_size_max << "\n";
  cout << "Result precision: " << result_precision << "\n\n";
}

void run() { run_stress_testing(); }

}  // namespace test_givens_rotator
