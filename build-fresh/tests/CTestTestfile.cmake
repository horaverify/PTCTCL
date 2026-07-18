# CMake generated Testfile for 
# Source directory: /Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests
# Build directory: /Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/build-fresh/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(version "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/build-fresh/ptctlmc" "--version")
set_tests_properties(version PROPERTIES  _BACKTRACE_TRIPLES "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;1;add_test;/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;0;")
add_test(selftest "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/build-fresh/ptctlmc" "--selftest")
set_tests_properties(selftest PROPERTIES  FAIL_REGULAR_EXPRESSION "failed" _BACKTRACE_TRIPLES "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;2;add_test;/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;0;")
add_test(unit "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/build-fresh/tests/ptctlmc_unit")
set_tests_properties(unit PROPERTIES  FAIL_REGULAR_EXPRESSION "FAIL" _BACKTRACE_TRIPLES "/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;7;add_test;/Users/brdlcu/Documents/Github/PTCTLMC/PTCTLMC/tests/CMakeLists.txt;0;")
