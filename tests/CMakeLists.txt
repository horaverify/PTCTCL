add_test(NAME version COMMAND ptctlmc --version)
add_test(NAME selftest COMMAND ptctlmc --selftest)
set_tests_properties(selftest PROPERTIES FAIL_REGULAR_EXPRESSION "failed")

add_executable(ptctlmc_unit dbm_tests.cpp)
target_link_libraries(ptctlmc_unit PRIVATE ptctlmc_core)
add_test(NAME unit COMMAND ptctlmc_unit)
set_tests_properties(unit PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")

add_executable(ptctlmc_bisim bisim_tests.cpp)
target_link_libraries(ptctlmc_bisim PRIVATE ptctlmc_core)
add_test(NAME bisim COMMAND ptctlmc_bisim)
set_tests_properties(bisim PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL ")
