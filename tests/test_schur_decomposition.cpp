#include <ctime>
#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/schur_decomposition.h"

namespace test_schur_decomposition {

using std::abs;
using std::cout;
using std::max;
using Algorithm = schur_decomposition::SchurDecomposition<double>;
using DynamicMatrix = Algorithm::DynamicMatrix;

constexpr const long double input_precision = 1e-10;
constexpr const long double result_precision = 1e-9;
constexpr const int number_of_tests = 100;
constexpr const int number_of_skipped_tests = 25;
constexpr const int matrix_size_max = 128;

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
                        (data - restored_data).norm());
    return false;
  }
  return true;
}

unsigned int time_check(int size) {
  DynamicMatrix data = DynamicMatrix::Random(size, size);
  DynamicMatrix result;
  DynamicMatrix backtrace;
  Algorithm algorithm(input_precision);
  unsigned int start_time = clock();
  algorithm.run(data, &result, &backtrace);
  unsigned int end_time = clock();
  return end_time - start_time;
}

void run_stress_testing() {
  for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
    for (int size = 1; size <= matrix_size_max; size *= 2) {
      srand(test_id);
      if (!simple_check(size, test_id)) return;
      if (!simple_check(size + 1, test_id)) return;
    }
  }

  cout << "Passed SchurDecomposition stress testing\n";
  cout << "Input precision: " << input_precision << "\n";
  cout << "Result precision: " << result_precision << "\n\n\n";
}

void measure_time(int matrix_size) {
  unsigned long long total_time = 0;
  double average_time;
  unsigned int max_time = 0;
  unsigned long long partial_test_total_time = 0;
  double partial_test_average_time;
  unsigned int partial_test_max_time = 0;

  for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
    srand(test_id);
    unsigned int time_checker = time_check(matrix_size);
    total_time += time_checker;
    max_time = max(max_time, time_checker);
    if (test_id > number_of_skipped_tests) {
      partial_test_total_time += time_checker;
      partial_test_max_time = max(partial_test_max_time, time_checker);
    }
  }

  average_time = (double)total_time / number_of_tests;
  partial_test_average_time = (double)partial_test_total_time /
                              (number_of_tests - number_of_skipped_tests);

  cout << "SchurDecomposition time measurement results with " << matrix_size
       << "x" << matrix_size << " matrices\n\n";
  cout << "Average time: " << average_time / CLOCKS_PER_SEC << "\n";
  cout << "Partial testing average time : "
       << partial_test_average_time / CLOCKS_PER_SEC << "\n";
  cout << "Max time: " << ((double)max_time) / CLOCKS_PER_SEC << "\n";
  cout << "Partial testing max time : "
       << ((double)partial_test_max_time) / CLOCKS_PER_SEC << "\n";
  cout << "Number of total tests: " << number_of_tests << "\n";
  cout << "Number of tests in partial testing: "
       << number_of_tests - number_of_skipped_tests << "\n";
}

void run() {
  cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
  run_stress_testing();
  measure_time(100);
  cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n";
}

}  // namespace test_schur_decomposition
