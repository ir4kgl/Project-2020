#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>

#include "../schur_decomposition/givens_rotation.h"

namespace test_find_angle {

using std::abs;
using std::arg;
using std::complex;
using std::cout;
using std::srand;
using std::time;

using AngleData = givens_rotation::AngleData<long double>;
using givens_rotation::find_angle;

constexpr const long double precision = 1e-18;
constexpr const int number_of_tests = 1000;

struct TestData {
  long double x;
  long double y;
  long double sin;
  long double cos;
  int id;
};

void process_check_failed(const TestData& test, const AngleData& angle) {
  cout << "test failed in function find_angle():\n\n";
  cout << "\tx = " << test.x << ";\t y = " << test.y << "\n";
  cout << "\texpected:\tcos = " << test.cos << ";";
  cout << "\t sin = " << test.sin << "\n";
  cout << "\tcalculated:\tcos = " << angle.cos << ";";
  cout << "\t sin = " << angle.sin << "\n\n";
  cout << "test id: " << test.id << "\n";
}

bool is_corresponding_angle(const TestData& test, const AngleData& angle) {
  return (abs(test.cos - angle.cos) < precision &&
          abs(test.sin - angle.sin) < precision);
}

bool simple_check(const TestData& test) {
  AngleData angle = find_angle<long double>(test.x, test.y);
  if (!is_corresponding_angle(test, angle)) {
    process_check_failed(test, angle);
    return false;
  }
  return true;
}

struct TestData generate_test(int test_id) {
  srand(test_id);
  long double x = (long double)(rand());
  long double y = (long double)(rand());
  complex<long double> value(x, y);
  long double angle = arg(value);
  return {x, y, std::sin(angle), std::cos(angle), test_id};
}

void run() {
  for (int test_id = 1; test_id <= number_of_tests; ++test_id) {
    TestData test = generate_test(test_id);
    if (!simple_check(test)) return;
  }

  cout << "Passed all find_angle() tests\n";
}

}  // namespace test_find_angle
