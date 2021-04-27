#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/hessenberg_reduction.h"

namespace test_hessenberg_reduction {

using std::cout;
using Algorithm = hessenberg_reduction::HessenbergReduction<double>;

constexpr const long double precision = 1e-12;
constexpr const int number_of_tests = 50;
constexpr const int matrix_size_max = 128;

bool is_hessenberg_form(const Algorithm::SquareMatrix& data) {
  return data.block(1, 0, data.rows() - 1, data.cols() - 1).isUpperTriangular();
}

bool are_equal(const Algorithm::SquareMatrix& first,
               const Algorithm::SquareMatrix& second) {
  return ((first - second).norm() < precision);
}

bool simple_check(int size, int test_id) {
  Algorithm::SquareMatrix data = Algorithm::SquareMatrix::Random(size, size);
  Algorithm::SquareMatrix old_data = data;
  Algorithm::UnitaryMatrix backtrace;

  Algorithm reduction;
  reduction.run(&data, &backtrace);

  if (!is_hessenberg_form(data)) {
    cout << "test failed in HessenbergReduction::run():\n\n";
    cout << "input: M =\n" << old_data << "\n\n";
    cout << "expected M' is upper Hessenberg form;\n\n";
    cout << "reduction to Hessenberg form result: M =\n" << data << "\n\n";
    cout << "test id:\t" << test_id << "\n";
    return false;
  }

  Algorithm::SquareMatrix restored_data =
      backtrace * data * backtrace.transpose();

  if (!are_equal(old_data, restored_data)) {
    cout << "test failed in HessenbergReduction::run(), wrong restore:\n\n";
    cout << "input: M =\n" << old_data << "\n\n";
    cout << "restored: M =\n" << restored_data << "\n\n";
    cout << "test id: " << test_id << "\n";
    return false;
  }
  return true;
}

void run() {
  for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
    for (int size = 4; size <= matrix_size_max; size *= 2) {
      srand(test_id);
      if (!simple_check(size, test_id)) return;
      if (!simple_check(size + 1, test_id)) return;
    }
  }

  cout << "Passed all HessenbergReduction tests\n";
}

}  // namespace test_hessenberg_reduction
