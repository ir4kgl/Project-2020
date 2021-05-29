#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/schur_decomposition.h"

namespace test_schur_decomposition {

using std::cout;
using std::max;
using Algorithm = schur_decomposition::SchurDecomposition<double>;
using DynamicMatrix = Algorithm::DynamicMatrix;
using DynamicVector = Algorithm::DynamicVector;

constexpr const double input_precision = 1e-12;
constexpr const double result_precision = 1e-11;
constexpr const int number_of_tests = 200;
constexpr const int number_of_skipped_tests = 50;
constexpr const int matrix_size_max = 200;

void process_unitary_check_failed(const DynamicMatrix& data,
                                  const DynamicMatrix& unitary, int test_id,
                                  int size) {
  cout << "test failed in SchurDecomposition::run():\n\n";
  cout << "input: M =\n" << data << "\n\n";
  cout << "expected unitary matrix;\n\n";
  cout << "calculated:\n" << unitary << "\n\n";
  cout << "size:\t" << size << "\n";
  cout << "test id:\t" << test_id << "\n";
}

void process_bad_restore(const DynamicMatrix& old_data,
                         const DynamicMatrix& restored_data, int test_id,
                         double delta, int size) {
  cout << "test failed in SchurDecomposition::run(), wrong restore:\n\n";
  cout << "input: M =\n" << old_data << "\n\n";
  cout << "restored: M =\n" << restored_data << "\n\n";
  cout << "delta = " << delta << "\n";
  cout << "size:\t" << size << "\n";
  cout << "test id: " << test_id << "\n";
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

bool simple_check(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(size, size);
  data += (DynamicMatrix)data.transpose();
  DynamicVector eigenvalues;
  DynamicMatrix backtrace;
  Algorithm algorithm(input_precision);
  algorithm.run(data, &eigenvalues, &backtrace);
  if (!backtrace.isUnitary(result_precision)) {
    process_unitary_check_failed(data, backtrace, test_id, size);
    return false;
  }
  DynamicMatrix restored_data = DynamicMatrix::Zero(size, size);
  restored_data.diagonal(0) = eigenvalues;
  restored_data = backtrace * restored_data * backtrace.transpose();
  if (!are_indistinguishable(data, restored_data, size)) {
    process_bad_restore(data, restored_data, test_id,
                        norm(data - restored_data, size), size);
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
