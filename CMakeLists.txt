cmake_minimum_required(VERSION 3.0.0)
project(QR VERSION 0.1.0)

set(TestSources tests/test_find_angle.cpp  tests/test_householder_reflector.cpp  tests/test_givens_rotator.cpp tests/test_hessenberg_reduction.cpp tests/tests.cpp)

add_executable(exec main.cpp "${TestSources}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
