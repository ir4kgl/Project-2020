cmake_minimum_required(VERSION 3.0.0)
project(QR VERSION 0.1.0)

include(CTest)
enable_testing()

set(TestSources tests/test_find_angle.cpp tests/tests.cpp)

add_executable(exec main.cpp "${TestSources}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
