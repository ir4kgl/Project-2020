#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/schur_decomposition.h"

namespace test_schur_decomposition {

using std::cout;
using Algorithm = schur_decomposition::SchurDecomposition<double>;
using DynamicMatrix = Algorithm::DynamicMatrix;

constexpr const long double precision = 1e-12;
constexpr const int number_of_tests = 50;
constexpr const int matrix_size_max = 128;

void process_triangular_check_failed(const DynamicMatrix& data,
                                     const DynamicMatrix& result, int test_id) {
  cout << "test failed in SchurDecomposition::run():\n\n";
  cout << "input: M =\n" << data << "\n\n";
  cout << "expected Schur form is quasi upper triangular matrix;\n\n";
  cout << "calculated Schur form:\n" << result << "\n\n";
  cout << "test id:\t" << test_id << "\n";
}

void process_bad_restore(const DynamicMatrix& old_data,
                         const DynamicMatrix& restored_data, int test_id,
                         double delta) {
  cout << "test failed in SchurDecomposition::run(), wrong restore:\n\n";
  // cout << "input: M =\n" << old_data << "\n\n";
  // cout << "restored: M =\n" << restored_data << "\n\n";
  cout << "delta = " << delta << "\n";
  cout << "test id: " << test_id << "\n";
}

bool is_hessenberg_form(const DynamicMatrix& result, int size) {
  return result.block(1, 0, size - 1, size - 1).isUpperTriangular(precision);
}

bool near_zero(double val) { return abs(val) < precision; }

bool are_indistinguishable(const DynamicMatrix& first,
                           const DynamicMatrix& second, int size) {
  return ((first - second).norm() < precision * size * size);
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

  Algorithm algorithm(precision);
  algorithm.run(data, &result, &backtrace);

  if (!is_quasi_triangular(result, size)) {
    process_triangular_check_failed(data, result, test_id);
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

void run() {
  for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
    for (int size = 1; size <= matrix_size_max; size *= 2) {
      srand(test_id);
      if (!simple_check(size, test_id)) return;
      if (!simple_check(size + 1, test_id)) return;
    }
  }

  cout << "Passed all SchurDecomposition tests\n";
}

}  // namespace test_schur_decomposition
