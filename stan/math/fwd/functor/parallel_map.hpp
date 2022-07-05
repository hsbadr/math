#ifndef STAN_MATH_FWD_FUNCTOR_PARALLEL_MAP_HPP
#define STAN_MATH_FWD_FUNCTOR_PARALLEL_MAP_HPP

#include <stan/math/prim/meta.hpp>

namespace stan {
namespace math {

/**
 * Evaluate a single-index loop (not parallelised)
 */
template <typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args,
          require_not_st_arithmetic<Res>* = nullptr,
          require_not_st_var<Res>* = nullptr>
inline void parallel_map_impl(const ApplyFunction& app_fun,
                              const IndexFunction& index_fun,
                              Res&& result, int grainsize, Args&&... x) {
  for (size_t i = 0; i < result.size(); ++i) {
    // Apply specified function to arguments at current iteration
    result[i] = index_fun(i, app_fun, x...);
  }
}

/**
 * Evaluate a two-index loop (not parallelised)
 */
template <typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args,
          require_not_st_arithmetic<Res>* = nullptr,
          require_not_st_var<Res>* = nullptr>
inline void parallel_map_impl(const ApplyFunction& app_fun,
                              const IndexFunction& index_fun,
                              Res&& result, int row_grainsize,
                              int col_grainsize, Args&&... x) {
  for (size_t j = 0; j < result.cols(); ++j) {
    for (size_t i = 0; i < result.rows(); ++i) {
      // Apply specified function to arguments at current iteration
      result(i, j) = index_fun(i, j, app_fun, x...);
    }
  }
}

}  // namespace math
}  // namespace stan
#endif
