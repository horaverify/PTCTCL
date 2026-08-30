/// @file hora/model_io.hpp
/// @brief Reading a model file into a hora::TimedAutomaton or hora::Network.
///
/// The input is UPPAAL's XML. It is the only model format HORA reads, so nothing downstream
/// depends on where a network came from: the reader is the single point that knows the format.
///
/// Supported: clocks, locations with invariants and urgent/committed flags, edges with guards,
/// clock resets and integer updates, bounded integer variables, and binary and broadcast channel
/// synchronisation. See docs/limitations.md for what is accepted but not fully honoured.
#pragma once

#include <stdexcept>
#include <string>

#include "hora/automaton.hpp"
#include "hora/network.hpp"

namespace hora {

/// @brief Thrown when a model file cannot be read or is not in the supported subset.
struct ModelError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// @brief Load a single-template model as one timed automaton.
/// @throws ModelError if the file cannot be read or declares more than one template.
TimedAutomaton load_model(const std::string& path);

/// @brief Load a multi-template model as a network.
/// @throws ModelError if the file cannot be read or uses an unsupported construct.
Network load_network(const std::string& path);

/// @brief How many processes the model's `system` declaration instantiates.
///
/// One template may be instantiated several times, so this is not the template count. Falls back
/// to the template count for models with no system list.
int network_process_count(const std::string& path);

}  // namespace hora
