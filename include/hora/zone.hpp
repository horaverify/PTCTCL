/// @file hora/zone.hpp
/// @brief HORA's view of the zone algebra: HDBM's names, imported into `hora`.
///
/// HDBM is a standalone library (`namespace hdbm`) with no dependency on this tool. Its zone
/// vocabulary — @c Dbm, @c Federation, @c Constraint, @c raw_t, the @c dbm_* and @c fed_*
/// operations — is used throughout the engine as if it were part of it, so it is imported here
/// once rather than qualified at several thousand call sites.
///
/// The import is deliberately scoped to `hora`: it cannot leak into a translation unit that has
/// not opened this namespace, and a consumer of HDBM alone still sees only `hdbm`.
///
/// Include this rather than the HDBM headers directly, unless a file genuinely needs only the
/// library and none of the engine.
#pragma once

#include "hdbm/constraint.hpp"
#include "hdbm/dbm.hpp"
#include "hdbm/dbm_ops.hpp"
#include "hdbm/dbm_pool.hpp"
#include "hdbm/fed.hpp"

namespace hora {
using namespace hdbm;  // NOLINT(google-build-using-namespace) — see the file comment
}  // namespace hora
