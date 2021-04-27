#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/householder_reflection.h"

namespace test_householder_reflector {

using std::cout;

using householder_reflection::HouseholderReflector;
using DynamicMatrix = HouseholderReflector<double>::DynamicMatrix;
using DynamicVector = HouseholderReflector<double>::DynamicVector;
using DynamicBlock = HouseholderReflector<double>::DynamicBlock;

constexpr const long double precision = 1e-14;
constexpr const int number_of_tests = 50;
constexpr const int matrix_size_max = 128;

void process_left_check_failed(const DynamicMatrix& data,
                               const DynamicMatrix& old_data, int test_id) {
  cout << "test failed in HouseholderReflector::reflect_left():\n\n";
  cout << "input: M = \n" << old_data << "\n\n";
  cout << "expected M(2, 0) ... M(N, 0) = 0;\n";
  cout << "reflection result: M = \n" << data << "\n\n";
  cout << "test id:\t" << test_id << "\n";
}

void process_right_check_failed(const DynamicMatrix& data,
                                const DynamicMatrix& old_data, int test_id) {
  cout << "test failed in HouseholderReflector::reflect_right():\n\n";
  cout << "input: M = \n" << old_data << "\n\n";
  cout << "expected M(0, 2) ... M(0, N) = 0;\n";
  cout << "reflection result: M = \n" << data << "\n\n";
  cout << "test id:\t" << test_id << "\n";
}

bool check_first_row(const DynamicMatrix& data, int size) {
  return (data.row(0).tail(size - 1).norm() < precision);
}

bool check_first_col(const DynamicMatrix& data, int size) {
  return (data.col(0).tail(size - 1).norm() < precision);
}

bool simple_check_left(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(size, size);
  DynamicMatrix old_data = data;
  HouseholderReflector reflector((DynamicVector)data.col(0));
  reflector.reflect_left(data.block(0, 0, size, size));

  if (!check_first_col(data, size)) {
    process_left_check_failed(data, old_data, test_id);
    return false;
  }
  return true;
}

bool simple_check_right(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(size, size);
  DynamicMatrix old_data = data;
  HouseholderReflector reflector((DynamicVector)data.row(0).transpose());
  reflector.reflect_right(data.block(0, 0, size, size));

  if (!check_first_row(data, size)) {
    process_right_check_failed(data, old_data, test_id);
    return false;
  }
  return true;
}

void run() {
  for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
    for (int size = 2; size <= matrix_size_max; size *= 2) {
      srand(test_id);
      if (!simple_check_left(size, test_id)) return;
      if (!simple_check_left(size + 1, test_id)) return;
      if (!simple_check_right(size, test_id)) return;
      if (!simple_check_right(size + 1, test_id)) return;
    }
  }

  cout << "Passed all HouseholderReflector tests\n";
}

}  // namespace test_householder_reflector
