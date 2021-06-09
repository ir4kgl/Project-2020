#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/householder_reflection.h"

namespace test_householder_reflector {

using std::cout;
using std::max;

using householder_reflection::HouseholderReflector;
using DynamicMatrix = HouseholderReflector<double>::DynamicMatrix;
using DynamicVector = HouseholderReflector<double>::DynamicVector;
using DynamicBlock = HouseholderReflector<double>::DynamicBlock;

constexpr const long double result_precision = 1e-14;
constexpr const int number_of_tests = 200;
constexpr const int matrix_size_max = 200;

void process_left_check_failed(const DynamicMatrix& data,
                               const DynamicMatrix& old_data, int test_id,
                               int size) {
  cout << "test failed in HouseholderReflector::reflect_left():\n\n";
  cout << "input: M = \n" << old_data << "\n\n";
  cout << "expected M(2, 0) ... M(N, 0) = 0;\n";
  cout << "reflection result: M = \n" << data << "\n\n";
  cout << "size:\t" << size << "\n";
  cout << "test id:\t" << test_id << "\n";
}

void process_right_check_failed(const DynamicMatrix& data,
                                const DynamicMatrix& old_data, int test_id,
                                int size) {
  cout << "test failed in HouseholderReflector::reflect_right():\n\n";
  cout << "input: M = \n" << old_data << "\n\n";
  cout << "expected M(0, 2) ... M(0, N) = 0;\n";
  cout << "reflection result: M = \n" << data << "\n\n";
  cout << "size:\t" << size << "\n";
  cout << "test id:\t" << test_id << "\n";
}

double norm(const DynamicMatrix& data) {
  double max_element = 0.;
  for (int i = 0; i < data.rows(); ++i) {
    for (int j = 0; j < data.cols(); ++j) {
      max_element = max(abs(data(i, j)), max_element);
    }
  }
  return max_element;
}

bool check_first_row(const DynamicMatrix& data, int size) {
  return (norm(data.row(0).tail(size - 1)) < result_precision);
}

bool check_first_col(const DynamicMatrix& data, int size) {
  return (norm(data.col(0).tail(size - 1)) < result_precision);
}

bool simple_check_left(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(size, size);
  DynamicMatrix old_data = data;
  HouseholderReflector reflector((DynamicVector)data.col(0));
  reflector.reflect_left(data.block(0, 0, size, size));
  if (!check_first_col(data, size)) {
    process_left_check_failed(data, old_data, test_id, size);
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
    process_right_check_failed(data, old_data, test_id, size);
    return false;
  }
  return true;
}

void run_stress_testing() {
  for (int size = 2; size <= matrix_size_max; ++size) {
    for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
      srand(test_id);
      if (!simple_check_left(size, test_id)) return;
      if (!simple_check_right(size, test_id)) return;
    }
  }
  cout << "Passed HouseholderReflector  stress testing\n";
  cout << "Number of tests: " << number_of_tests << "\n";
  cout << "Maximum matrix size: " << matrix_size_max << "\n";
  cout << "Result precision: " << result_precision << "\n\n\n";
}

void run() { run_stress_testing(); }

}  // namespace test_householder_reflector
