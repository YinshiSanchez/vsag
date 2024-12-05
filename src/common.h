
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <unordered_map>

#define SAFE_CALL(stmt)                                                              \
    try {                                                                            \
        stmt;                                                                        \
    } catch (const std::exception& e) {                                              \
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknownError: ", e.what()); \
    } catch (...) {                                                                  \
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknown error");            \
    }

#define CHECK_ARGUMENT(expr, message)             \
    do {                                          \
        if (not(expr)) {                          \
            throw std::invalid_argument(message); \
        }                                         \
    } while (0);

#define ROW_ID_MASK 0xFFFFFFFFLL

namespace vsag {
namespace glass {
enum class Metric {
    L2,
    IP,
};

inline std::unordered_map<std::string, Metric> metric_map;

inline int metric_map_init = [] {
    metric_map["L2"] = Metric::L2;
    metric_map["IP"] = Metric::IP;
    return 42;
}();

inline constexpr size_t
upper_div(size_t x, size_t y) {
    return (x + y - 1) / y;
}

inline constexpr int64_t
do_align(int64_t x, int64_t align) {
    return (x + align - 1) / align * align;
}

#if defined(__clang__)

#define FAST_BEGIN
#define FAST_END
#define GLASS_INLINE __attribute__((always_inline))

#elif defined(__GNUC__)

#define FAST_BEGIN              \
    _Pragma("GCC push_options") \
        _Pragma("GCC optimize (\"unroll-loops,associative-math,no-signed-zeros\")")
#define FAST_END _Pragma("GCC pop_options")
#define GLASS_INLINE [[gnu::always_inline]]
#else

#define FAST_BEGIN
#define FAST_END
#define GLASS_INLINE

#endif

}  // namespace glass

}  // namespace vsag
