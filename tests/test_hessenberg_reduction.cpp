#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/hessenberg_reduction.h"

namespace test_hessenberg_reduction {

using std::cout;
using std::max;
using Algorithm = hessenberg_reduction::HessenbergReduction<double>;
using DynamicMatrix = Algorithm::DynamicMatrix;

constexpr const long double result_precision = 1e-11;
constexpr const int number_of_tests = 200;
constexpr const int matrix_size_max = 200;

bool is_hessenberg_form(const DynamicMatrix& data, int size) {
  return data.block(1, 0, size - 1, size - 1)
      .isUpperTriangular(result_precision);
}

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

void process_hessenberg_check_failed(const DynamicMatrix& data,
                                     const DynamicMatrix& old_data, int test_id,
                                     int size) {
  cout << "test failed in HessenbergReduction::run():\n\n";
  cout << "input: M =\n" << old_data << "\n\n";
  cout << "expected M is upper Hessenberg form;\n\n";
  cout << "reduction to Hessenberg form result: M =\n" << data << "\n\n";
  cout << "size:\t" << size << "\n";
  cout << "test id:\t" << test_id << "\n";
}

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
                         int size) {
  cout << "test failed in HessenbergReduction::run(), wrong restore:\n\n";
  cout << "input: M =\n" << old_data << "\n\n";
  cout << "restored: M =\n" << restored_data << "\n\n";
  cout << "size:\t" << size << "\n";
  cout << "test id:\n" << test_id << "\n";
}

bool simple_check(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(size, size);
  DynamicMatrix old_data = data;
  DynamicMatrix backtrace;
  Algorithm reduction;
  reduction.run(&data, &backtrace);
  if (!is_hessenberg_form(data, size)) {
    process_hessenberg_check_failed(data, old_data, test_id, size);
    return false;
  }
  if (!backtrace.isUnitary()) {
    process_unitary_check_failed(data, backtrace, test_id, size);
    return false;
  }
  DynamicMatrix restored_data = backtrace * data * backtrace.transpose();
  if (!are_indistinguishable(old_data, restored_data, size)) {
    process_bad_restore(old_data, restored_data, test_id, size);
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

  cout << "Passed HessenbergReduction stress testing\n";
  cout << "Number of tests: " << number_of_tests << "\n";
  cout << "Maximum matrix size: " << matrix_size_max << "\n";
  cout << "Result precision: " << result_precision << "\n\n\n";
}

void run() { run_stress_testing(); }

}  // namespace test_hessenberg_reduction
