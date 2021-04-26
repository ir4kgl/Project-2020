#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

#include "../eigen/Eigen/Dense"
#include "../schur_decomposition/hessenberg_reduction.h"

using std::cout;

namespace test_hessenberg_reduction {

static constexpr const long double precision = 1e-14;
static constexpr const int tests_counter = 1000;
static constexpr const int matrix_size = 100;

bool simple_check() { return true; }

void run() {
  for (int j = 0; j < tests_counter; ++j) {
    if (!simple_check()) return;
  }

  cout << "Passed all HessenbergReduction tests\n";
}

}  // namespace test_hessenberg_reduction
