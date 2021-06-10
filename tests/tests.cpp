#include "test_givens_rotator.h"
#include "test_hessenberg_reduction.h"
#include "test_householder_reflector.h"
#include "test_schur_decomposition.h"
#include "test_schur_decomposition_symmetric.h"

void run_all_tests() {
  test_givens_rotator::run();
  test_householder_reflector::run();
  test_hessenberg_reduction::run();
  test_schur_decomposition::run();
  test_schur_decomposition_symmetric::run();
}
