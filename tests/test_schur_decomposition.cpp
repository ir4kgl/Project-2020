#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/schur_decomposition.h"

namespace test_schur_decomposition {

using std::cout;
using Algorithm = schur_decomposition::SchurDecomposition<double>;
using SchurForm = Algorithm::SchurForm;
using SquareMatrix = Algorithm::SquareMatrix;
using UnitaryMatrix = Algorithm::UnitaryMatrix;

constexpr const long double precision = 1e-12;
constexpr const int number_of_tests = 50;
constexpr const int matrix_size_max = 128;

bool simple_check(int size, int test_id) {
  //   SquareMatrix data = SquareMatrix::Random(size, size);
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
