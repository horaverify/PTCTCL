/// @file disc_key.hpp
/// @brief Key types for the packed discrete state, and the mixing they share.
///
/// The forward search keys its passed set on a lossless bit-packing of the discrete state (the
/// location vector plus the integer variables). One machine word holds that packing for every
/// network the benchmarks care about, and a word-sized key is what makes the passed set's slot
/// array, the stripe index and the per-level frontier cheap.
///
/// A wide network does not fit. Rather than reject it -- or pay a wider key everywhere -- the
/// engine is a template over the key type: ::hora::detail::NarrowKey is the default and is
/// unchanged, and ::hora::detail::WideKey is instantiated only for a network whose fields need
/// more than ::hora::detail::key_budget<NarrowKey>() bits. Nothing about the narrow path changes,
/// so a table measured before this existed stays comparable with one measured after.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace hora {
namespace detail {

/// @brief The default packed-key type: one machine word.
using NarrowKey = std::uint64_t;

/// @brief The fallback packed-key type, for a network too wide for ::NarrowKey.
///
/// @note `unsigned __int128` is a compiler extension, but one both GCC and Clang provide on every
///       platform this project builds for. It is used only as a bit container -- shifted, masked
///       and compared -- so it needs no library support beyond what the language gives it.
using WideKey = unsigned __int128;

/// @brief Bits a key type can actually carry as a packed discrete state.
///
/// One bit short of the type's width: ::DiscKeyCodec requires every field offset to be strictly
/// below the width, so that a trailing zero-width field is never shifted by exactly the width,
/// which is undefined.
template <class Key>
constexpr int key_budget() {
  return static_cast<int>(sizeof(Key) * 8) - 1;
}

/// @brief Multiplicative mix of a packed key, for table and stripe indexing.
///
/// @param dk the packed discrete key
/// @return a 64-bit hash whose HIGH bits are well mixed
///
/// @note Callers take the high bits (`>> shift`), which is what makes plain Fibonacci
///       multiplication sufficient: the top of the product depends on every input bit.
inline std::uint64_t key_mix(NarrowKey dk) {
  return dk * 0x9e3779b97f4a7c15ULL;
}

/// @brief Mix for the wide key: fold the halves, then mix as usual.
///
/// @param dk the packed discrete key
/// @return a 64-bit hash whose high bits depend on both halves of @p dk
///
/// @note The fold is an xor rather than an add so that it cannot lose the high half to a carry
///       chain, and the halves are multiplied apart first because a network wide enough to need
///       this key often leaves one half sparsely populated -- xoring the raw halves of two such
///       keys collides far too readily.
inline std::uint64_t key_mix(WideKey dk) {
  const std::uint64_t lo = static_cast<std::uint64_t>(dk);
  const std::uint64_t hi = static_cast<std::uint64_t>(dk >> 64);
  return (lo * 0x9e3779b97f4a7c15ULL) ^ (hi * 0xc2b2ae3d27d4eb4fULL);
}

/// @brief Hash functor for a packed key, so it can be an `unordered_map` key.
///
/// @note ::WideKey has no `std::hash` specialisation, and adding one would be an intrusion into
///       the standard namespace for a type the standard does not own. A local functor is the
///       narrower fix, and it keeps both key types on the same mixing function.
struct KeyHash {
  template <class Key>
  std::size_t operator()(Key dk) const {
    return static_cast<std::size_t>(key_mix(dk));
  }
};

/// @brief The hash an unordered container should use for a packed key.
///
/// @warning This must stay `std::hash` for ::NarrowKey. The forward search iterates the touched
///          set to build the next frontier, so the container's bucket order is part of the
///          exploration order, and the exploration order decides which of two mutually subsuming
///          zones is stored first. Changing the hash therefore changes recorded STATE COUNTS on a
///          parallel run -- not the verdict, but enough to make an existing measurement
///          irreproducible. Measured on grc4 (714 -> 765), grc6 and phil8 when this was briefly
///          ::KeyHash for both key types.
template <class Key>
struct KeyHashFor {
  using type = std::hash<Key>;
};

/// @note ::WideKey is a compiler extension, so a standard hash for it is not guaranteed. It is
///       also new, with no existing measurement to preserve, so the fold is free to be used here.
template <>
struct KeyHashFor<WideKey> {
  using type = KeyHash;
};

}  // namespace detail
}  // namespace hora
