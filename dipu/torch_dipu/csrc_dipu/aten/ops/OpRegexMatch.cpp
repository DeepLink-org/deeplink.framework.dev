// Copyright (c) 2024, DeepLink.
#include "OpRegexMatch.hpp"

#include <algorithm>
#include <fstream>
#include <ios>
#include <iostream>
#include <regex>

#include <c10/util/Exception.h>

// loadMatcher is used to get regex matcher from env_name and config
// kFallbackEnvName = "DIPU_FORCE_FALLBACK_OPS_LIST"; kFallbackConfigName =
// ".dipu_force_fallback_op_list.config" kSpecifiedAutocompareEnvName =
// "DIPU_AUTOCOMPARE_OPS_LIST"; kSpecifiedAutocompareConfigName =
// ".specified_autocompare_op_list.config"

namespace dipu {
namespace op_regex_match {
std::vector<std::regex> loadMatcher(const char* env_name,
                                    const char* config_name) {
  auto append = [](std::istream& input, std::vector<std::regex>& output) {
    auto constexpr separator = ',';

    auto line = std::string();
    while (std::getline(input, line)) {
      auto buffer = std::istringstream(line);
      auto pattern = std::string();
      while (std::getline(buffer, pattern, separator)) {
        if (pattern.empty()) {
          continue;
        }
        try {
          output.emplace_back(pattern);
        } catch (const std::regex_error& e) {
          TORCH_CHECK(false, e.what());
        }
      }
    }
  };

  auto list = std::vector<std::regex>();
  if (auto env = std::getenv(env_name)) {
    auto iss = std::istringstream(env);
    append(iss, list);
  }
  if (auto file = std::ifstream(config_name, std::ios::binary)) {
    append(file, list);
  }
  return list;
}

bool isOpMatch(const char* opname,
               const std::vector<std::regex>& regexMatchers) {
  if (regexMatchers.empty() || opname == nullptr) {
    return false;
  }

  return std::any_of(
      regexMatchers.begin(), regexMatchers.end(),
      [&opname](auto& matcher) { return std::regex_match(opname, matcher); });
}

constexpr const char* kFallbackEnvName = "DIPU_FORCE_FALLBACK_OPS_LIST";
constexpr const char* kFallbackConfigName =
    ".dipu_force_fallback_op_list.config";
const std::vector<std::regex> kFallbackMatchers =
    dipu::op_regex_match::loadMatcher(kFallbackEnvName, kFallbackConfigName);

constexpr const char* kSpecifiedAutocompareEnvName =
    "DIPU_AUTOCOMPARE_OPS_LIST";
constexpr const char* kSpecifiedAutocompareConfigName =
    ".specified_autocompare_op_list.config";
const std::vector<std::regex> kAutocompareMatchers =
    dipu::op_regex_match::loadMatcher(kSpecifiedAutocompareEnvName,
                                      kSpecifiedAutocompareConfigName);
}  // namespace op_regex_match
}  // namespace dipu
