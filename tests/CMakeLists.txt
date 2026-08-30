add_test(NAME version COMMAND hora --version)
add_test(NAME selftest COMMAND hora --selftest)
set_tests_properties(selftest PROPERTIES FAIL_REGULAR_EXPRESSION "failed")

add_executable(hora_unit dbm_tests.cpp)
target_link_libraries(hora_unit PRIVATE hora_core)
add_test(NAME unit COMMAND hora_unit)
set_tests_properties(unit PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")

# Measurement driver for the environment-abstraction lead (not a ctest: it is a study, and the
# larger instances are deliberately long-running). docs/superpowers/specs/weaksim-quotient.md
add_executable(hora_envquot env_quotient.cpp)
target_link_libraries(hora_envquot PRIVATE hora_core)

# The zone-algebra property tests live in HDBM, next to the ops they pin down (dbm_free and
# dbm_down claim to need no closure; dbm_try_union claims its minimal-graph analysis prunes work
# and never an answer). HDBM only registers its own tests when it is the top-level project, so
# build the same source here to keep it in this project's ctest run.
add_executable(hdbm_props ${CMAKE_CURRENT_SOURCE_DIR}/../HDBM/test/dbm_ops_props_test.cpp)
target_link_libraries(hdbm_props PRIVATE hdbm)
add_test(NAME hdbm_props COMMAND hdbm_props)
set_tests_properties(hdbm_props PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL")

add_executable(hora_bisim bisim_tests.cpp)
target_link_libraries(hora_bisim PRIVATE hora_core)
add_test(NAME bisim COMMAND hora_bisim)
set_tests_properties(bisim PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL ")
