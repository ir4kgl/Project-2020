#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Eigenvalues"
#include "../schur_decomposition/schur_decomposition.h"

namespace test_schur_decomposition {

using std::abs;
using std::cout;
using std::max;
using std::min;
using Algorithm = schur_decomposition::SchurDecomposition<double>;
using DynamicMatrix = Algorithm::DynamicMatrix;

constexpr const long double input_precision = 1e-12;
constexpr const long double result_precision = 1e-11;
constexpr const int number_of_tests = 100;
constexpr const int number_of_skipped_tests = 25;
constexpr const int matrix_size_max = 100;

void process_triangular_check_failed(const DynamicMatrix& data,
                                     const DynamicMatrix& result, int test_id) {
  cout << "test failed in SchurDecomposition::run():\n\n";
  cout << "input: M =\n" << data << "\n\n";
  cout << "expected Schur form is quasi upper triangular matrix;\n\n";
  cout << "calculated Schur form:\n" << result << "\n\n";
  cout << "test id:\t" << test_id << "\n";
}

void process_unitary_check_failed(const DynamicMatrix& data,
                                  const DynamicMatrix& unitary, int test_id) {
  cout << "test failed in SchurDecomposition::run():\n\n";
  cout << "input: M =\n" << data << "\n\n";
  cout << "expected unitary matrix;\n\n";
  cout << "calculated:\n" << unitary << "\n\n";
  cout << "test id:\t" << test_id << "\n";
}

void process_bad_restore(const DynamicMatrix& old_data,
                         const DynamicMatrix& restored_data, int test_id,
                         double delta) {
  cout << "test failed in SchurDecomposition::run(), wrong restore:\n\n";
  cout << "input: M =\n" << old_data << "\n\n";
  cout << "restored: M =\n" << restored_data << "\n\n";
  cout << "delta = " << delta << "\n";
  cout << "test id: " << test_id << "\n";
}

bool is_hessenberg_form(const DynamicMatrix& result, int size) {
  return result.block(1, 0, size - 1, size - 1)
      .isUpperTriangular(result_precision);
}

bool near_zero(double val) { return abs(val) < result_precision; }

double norm(const DynamicMatrix& data, int size) {
  double max_element = 0.;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j)
      max_element = max(abs(data(i, j)), max_element);
  }
  return max_element;
}

bool are_indistinguishable(const DynamicMatrix& first,
                           const DynamicMatrix& second, int size) {
  return (norm(first - second, size) < result_precision);
}

bool is_quasi_triangular(const DynamicMatrix& result, int size) {
  if (!is_hessenberg_form(result, size)) return false;
  for (int j = 1; j < size - 1; ++j) {
    if (!near_zero(result(j + 1, j) && !near_zero(result(j, j - 1)))) {
      return false;
    }
  }
  return true;
}

bool simple_check(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(size, size);
  DynamicMatrix result;
  DynamicMatrix backtrace;
  Algorithm algorithm(input_precision);
  algorithm.run(data, &result, &backtrace);
  if (!is_quasi_triangular(result, size)) {
    process_triangular_check_failed(data, result, test_id);
    return false;
  }
  if (!backtrace.isUnitary()) {
    process_unitary_check_failed(data, backtrace, test_id);
    return false;
  }
  DynamicMatrix restored_data = backtrace * result * backtrace.transpose();
  if (!are_indistinguishable(data, restored_data, size)) {
    process_bad_restore(data, restored_data, test_id,
                        norm(data - restored_data, size));
    return false;
  }
  return true;
}

void run_stress_testing() {
  for (int size = 1; size <= matrix_size_max; ++size) {
    for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
      srand(test_id);
      if (!simple_check(size, test_id)) return;
    }
  }

  cout << "Passed SchurDecomposition stress testing\n";
  cout << "Number of tests: " << number_of_tests << "\n";
  cout << "Maximum matrix size: " << matrix_size_max << "\n";
  cout << "Input precision: " << input_precision << "\n";
  cout << "Result precision: " << result_precision << "\n\n\n";
}

void run() { run_stress_testing(); }

}  // namespace test_schur_decomposition
