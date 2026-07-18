# CMake generated Testfile for 
# Source directory: /home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/tests
# Build directory: /home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(version "/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/build/ptctlmc" "--version")
set_tests_properties(version PROPERTIES  _BACKTRACE_TRIPLES "/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/tests/CMakeLists.txt;1;add_test;/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/tests/CMakeLists.txt;0;")
add_test(selftest "/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/build/ptctlmc" "--selftest")
set_tests_properties(selftest PROPERTIES  FAIL_REGULAR_EXPRESSION "failed" _BACKTRACE_TRIPLES "/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/tests/CMakeLists.txt;2;add_test;/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/tests/CMakeLists.txt;0;")
add_test(unit "/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/build/tests/ptctlmc_unit")
set_tests_properties(unit PROPERTIES  FAIL_REGULAR_EXPRESSION "FAIL" _BACKTRACE_TRIPLES "/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/tests/CMakeLists.txt;7;add_test;/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/tests/CMakeLists.txt;0;")
add_test(bisim "/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/build/tests/ptctlmc_bisim")
set_tests_properties(bisim PROPERTIES  FAIL_REGULAR_EXPRESSION "FAIL " _BACKTRACE_TRIPLES "/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/tests/CMakeLists.txt;12;add_test;/home/luca/Documents/ParallelCTLModelChecker/PTCTLMC/tests/CMakeLists.txt;0;")
