#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/hessenberg_reduction.h"

namespace test_hessenberg_reduction {

using std::cout;
using Algorithm = hessenberg_reduction::HessenbergReduction<double>;
using DynamicMatrix = Algorithm::DynamicMatrix;

constexpr const long double precision = 1e-12;
constexpr const int number_of_tests = 50;
constexpr const int matrix_size_max = 128;

bool is_hessenberg_form(const DynamicMatrix& data, int size) {
  return data.block(1, 0, size - 1, size - 1).isUpperTriangular(precision);
}

bool are_almost_equal(const DynamicMatrix& first, const DynamicMatrix& second) {
  return ((first - second).norm() < precision);
}

void process_hessenberg_check_failed(const DynamicMatrix& data,
                                     const DynamicMatrix& old_data,
                                     int test_id) {
  cout << "test failed in HessenbergReduction::run():\n\n";
  cout << "input: M =\n" << old_data << "\n\n";
  cout << "expected M is upper Hessenberg form;\n\n";
  cout << "reduction to Hessenberg form result: M =\n" << data << "\n\n";
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
                         const DynamicMatrix& restored_data, int test_id) {
  cout << "test failed in HessenbergReduction::run(), wrong restore:\n\n";
  cout << "input: M =\n" << old_data << "\n\n";
  cout << "restored: M =\n" << restored_data << "\n\n";
  cout << "test id: " << test_id << "\n";
}

bool simple_check(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(size, size);
  DynamicMatrix old_data = data;
  DynamicMatrix backtrace;

  Algorithm reduction;
  reduction.run(&data, &backtrace);

  if (!is_hessenberg_form(data, size)) {
    process_hessenberg_check_failed(data, old_data, size);
    return false;
  }

  if (!backtrace.isUnitary()) {
    process_unitary_check_failed(data, backtrace, test_id);
    return false;
  }

  DynamicMatrix restored_data = backtrace * data * backtrace.transpose();
  if (!are_almost_equal(old_data, restored_data)) {
    process_bad_restore(old_data, restored_data, test_id);
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

  cout << "Passed all HessenbergReduction tests\n";
}

}  // namespace test_hessenberg_reduction
