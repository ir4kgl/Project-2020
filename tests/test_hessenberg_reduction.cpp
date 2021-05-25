#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/hessenberg_reduction.h"

namespace test_hessenberg_reduction {

using std::cout;
using Algorithm = hessenberg_reduction::HessenbergReduction<double>;
using DynamicMatrix = Algorithm::DynamicMatrix;

constexpr const long double precision = 1e-15;
constexpr const int number_of_tests = 50;
constexpr const int matrix_size_max = 128;

bool are_indistinguishable(const DynamicMatrix& first,
                           const DynamicMatrix& second, int size) {
  return ((first - second).norm() < precision * size * size);
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
  cout << "delta: " << (old_data - restored_data).norm() << "\n";
  cout << "test id: " << test_id << "\n";
}

bool simple_check(int size, int test_id) {
  DynamicMatrix data = DynamicMatrix::Random(size, size);
  data *= data.transpose();
  DynamicMatrix backtrace;

  Algorithm reduction;
  DynamicMatrix old_data = data;
  reduction.run(&data, &backtrace);

  if (!backtrace.isUnitary()) {
    process_unitary_check_failed(data, backtrace, test_id);
    return false;
  }
  DynamicMatrix restored_data = backtrace * data * backtrace.transpose();
  if (!are_indistinguishable(old_data, restored_data, size)) {
    process_bad_restore(data, restored_data, test_id);
    return false;
  }

  return true;
}

void run() {
  for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
    for (int size = 2; size <= matrix_size_max; size *= 2) {
      srand(test_id);
      if (!simple_check(size, test_id)) return;
      if (!simple_check(size + 1, test_id)) return;
    }
  }

  cout << "Passed all HessenbergReduction tests\n";
}

}  // namespace test_hessenberg_reduction
