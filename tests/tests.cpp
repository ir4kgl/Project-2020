#include "test_find_angle.h"
#include "test_givens_rotator.h"
#include "test_hessenberg_reduction.h"
#include "test_householder_reflector.h"

void run_all_tests() {
  test_find_angle::run();
  test_givens_rotator::run();
  test_householder_reflector::run();
  test_hessenberg_reduction::run();
}
