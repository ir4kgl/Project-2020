#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/hessenberg_reduction.h"

using std::cout;

using Eigen::MatrixXd;
using hessenberg_reduction::HessenbergReduction;

namespace test_hessenberg_reduction {

static constexpr const long double precision = 1e-12;
static constexpr const int tests_counter = 300;
static constexpr const int matrix_size = 100;

bool simple_check() {
  MatrixXd data = MatrixXd::Random(matrix_size, matrix_size);
  MatrixXd old_data = data;
  MatrixXd backtrace;

  HessenbergReduction<double> reduction;
  reduction.run(&data, &backtrace);

  if (!data.block(1, 0, matrix_size - 1, matrix_size - 1).isUpperTriangular()) {
    cout << "test failed in HessenbergReduction::run():\n\n";
    cout << "input: M =\n" << old_data << "\n\n";
    cout << "expected M' is upper Hessenberg form;\n\n";
    cout << "reduction to Hessenberg form result:\n" << data << "\n";
    return false;
  }

  MatrixXd restored_data = backtrace * data * backtrace.transpose();

  if ((old_data - restored_data).norm() > precision) {
    cout << "test failed in HessenbergReduction::run(), wrong restore:\n\n";
    cout << "input: M =\n" << old_data << "\n\n";
    cout << "restored M:\n" << restored_data << "\n";
    return false;
  }
  return true;
}

void run() {
  for (int j = 0; j < tests_counter; ++j) {
    if (!simple_check()) return;
  }

  cout << "Passed all HessenbergReduction tests\n";
}

}  // namespace test_hessenberg_reduction
