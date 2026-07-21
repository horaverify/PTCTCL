# CMake generated Testfile for 
# Source directory: /Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests
# Build directory: /Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/build-omp/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(version "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/build-omp/ptctlmc" "--version")
set_tests_properties(version PROPERTIES  _BACKTRACE_TRIPLES "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;1;add_test;/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;0;")
add_test(selftest "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/build-omp/ptctlmc" "--selftest")
set_tests_properties(selftest PROPERTIES  FAIL_REGULAR_EXPRESSION "failed" _BACKTRACE_TRIPLES "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;2;add_test;/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;0;")
add_test(unit "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/build-omp/tests/ptctlmc_unit")
set_tests_properties(unit PROPERTIES  FAIL_REGULAR_EXPRESSION "FAIL" _BACKTRACE_TRIPLES "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;7;add_test;/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;0;")
add_test(bisim "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/build-omp/tests/ptctlmc_bisim")
set_tests_properties(bisim PROPERTIES  FAIL_REGULAR_EXPRESSION "FAIL " _BACKTRACE_TRIPLES "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;12;add_test;/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;0;")
