#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>

#include "../schur_decomposition/givens_rotation.h"

using std::abs;
using std::arg;
using std::complex;
using std::cout;
using std::srand;
using std::time;

using givens_rotation::AngleData;
using givens_rotation::find_angle;

namespace test_find_angle {

struct TestData {
  long double x;
  long double y;
  long double sin;
  long double cos;
};

static constexpr const long double precision = 1e-18;
static constexpr const int tests_counter = 1000;

bool simple_check(const TestData& test) {
  AngleData<long double> angle = find_angle<long double>(test.x, test.y);

  if (abs(test.cos - angle.cos) > precision ||
      abs(test.sin - angle.sin) > precision) {
    cout << "test failed in function find_angle():\n\n";
    cout << "\tx = " << test.x << ";\t y = " << test.y << "\n";
    cout << "\texpected:\tcos = " << test.cos << ";";
    cout << "\t sin = " << test.sin << "\n";
    cout << "\tcalculated:\tcos = " << angle.cos << ";";
    cout << "\t sin = " << angle.sin << "\n";
    return false;
  }

  return true;
}

struct TestData generate_test() {
  srand(time(nullptr));
  long double x = (long double)(rand());
  long double y = (long double)(rand());
  complex<long double> value(x, y);
  long double angle = arg(value);
  return {x, y, std::sin(angle), std::cos(angle)};
}

void run() {
  for (int j = 0; j < tests_counter; ++j) {
    TestData test = generate_test();
    if (!simple_check(test)) return;
  }

  cout << "Passed all find_angle() tests\n";
}

}  // namespace test_find_angle
