#ifndef STAN_MATH_PRIM_FUNCTOR_PARALLEL_MAP_HPP
#define STAN_MATH_PRIM_FUNCTOR_PARALLEL_MAP_HPP

#include <stan/math/prim/meta.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>

namespace stan {
namespace math {

/**
 * Evaluate a single-index loop in parallel
 */
template <typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args,
          require_st_arithmetic<Res>* = nullptr>
inline void parallel_map_impl(const ApplyFunction& app_fun,
                         const IndexFunction& index_fun,
                         Res&& result, int grainsize, Args&&... x) {
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, result.size(), grainsize),
    [&](
     const tbb::blocked_range<size_t>& r) {
      for (size_t i = r.begin(); i < r.end(); ++i) {
        // Apply specified function to arguments at current iteration
        result[i] = index_fun(i, app_fun, x...);
      }
    });
}

/**
 * Evaluate a two-index loop in parallel
 */
template <typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args,
          require_st_arithmetic<Res>* = nullptr>
inline void parallel_map_impl(const ApplyFunction& app_fun,
                         const IndexFunction& index_fun,
                         Res&& result, int row_grainsize,
                         int col_grainsize, Args&&... x) {
  tbb::parallel_for(
    tbb::blocked_range2d<size_t>(0, result.rows(), row_grainsize,
                                 0, result.cols(), col_grainsize),
    [&](
     const tbb::blocked_range2d<size_t>& r) {
      for (size_t j = r.cols().begin(); j < r.cols().end(); ++j) {
        for (size_t i = r.rows().begin(); i < r.rows().end(); ++i) {
          // Apply specified function to arguments at current iteration
          result(i, j) = index_fun(i, j, app_fun, x...);
        }
      }
    });
}


template <typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args>
inline void parallel_map(const ApplyFunction& app_fun,
                         const IndexFunction& index_fun,
                         Res&& result, int grainsize, Args&&... x) {
#ifdef STAN_THREADS
  parallel_map_impl(app_fun, index_fun, std::forward<Res>(result), grainsize,
                    std::forward<Args>(x)...);
#else
  for (size_t i = 0; i < result.size(); ++i) {
    // Apply specified function to arguments at current iteration
    result[i] = index_fun(i, app_fun, x...);
  }
#endif
}

template <typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args>
inline void parallel_map(const ApplyFunction& app_fun,
                         const IndexFunction& index_fun,
                         Res&& result, int row_grainsize,
                         int col_grainsize, Args&&... x) {
#ifdef STAN_THREADS
  parallel_map_impl(app_fun, index_fun, std::forward<Res>(result),
                    row_grainsize, col_grainsize, std::forward<Args>(x)...);
#else
  for (size_t j = 0; j < result.cols(); ++j) {
    for (size_t i = 0; i < result.rows(); ++i) {
      // Apply specified function to arguments at current iteration
      result(i, j) = index_fun(i, j, app_fun, x...);
    }
  }
#endif
}

}  // namespace math
}  // namespace stan
#endif
