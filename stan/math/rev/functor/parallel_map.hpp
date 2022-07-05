#ifndef STAN_MATH_REV_FUNCTOR_PARALLEL_MAP_HPP
#define STAN_MATH_REV_FUNCTOR_PARALLEL_MAP_HPP

#include <stan/math/rev/core.hpp>
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
          require_st_var<Res>* = nullptr>
inline void parallel_map_impl(const ApplyFunction& app_fun,
                              const IndexFunction& index_fun,
                              Res&& result, int grainsize,
                              Args&&... x) {
    // Functors for manipulating vars at a given iteration of the loop
    auto var_counter = [&](auto&... xargs) {
      return count_vars(xargs...);
    };
    auto var_copier = [&](const auto&... xargs) {
      return std::tuple<decltype(deep_copy_vars(xargs))...>(
        deep_copy_vars(xargs)...);
    };
    auto vari_saver = [&](vari** varis) {
      return [=](const auto&... xargs) {
        save_varis(varis, xargs...);
      };
    };

    int S = result.size();

    // Assuming that the number of the vars at each iteration of the loop is
    // the same (as the operations at each iteration should be the same), we can
    // just count vars at the first iteration.
    int nvars = index_fun(0, var_counter, x...);

    // Allocate memory for varis, values, and partials
    vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
      S * nvars);
    double* values = ChainableStack::instance_->memalloc_.alloc_array<double>(
      S);
    double* partials = ChainableStack::instance_->memalloc_.alloc_array<double>(
      S * nvars);

    // By using a Map with an InnerStride of length nvars we can index the Map
    // normally (i.e., from 0 to S-1) to get the location in memory for the
    // varis/adjoints for that iteration. Where the InnerStride (nvars) is the
    // distance between consecutive indexes.
    Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<>> par_map(
      partials, S, Eigen::InnerStride<>(nvars));
    Eigen::Map<stan::math::vector_vi, 0, Eigen::InnerStride<>> vari_map(
      varis, S, Eigen::InnerStride<>(nvars));

    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, S, grainsize),
      [&](const tbb::blocked_range<size_t>& r) {
        // Run nested autodiff in this scope
        const nested_rev_autodiff nested;

        for (size_t i = r.begin(); i < r.end(); ++i) {
          index_fun(i, vari_saver(&vari_map[i]), x...);
          auto args_tuple_local_copy = index_fun(i, var_copier, x...);

          // Apply specified function to arguments at current iteration
          var out = apply([&](auto&&... args) { return app_fun(args...); },
                          args_tuple_local_copy);

          out.grad();

          // Extract value and adjoints to be put into vars on main
          // autodiff stack
          values[i] = std::move(out.vi_->val_);
          apply([&](auto&&... args) {
              save_adjoints(&par_map[i],
                            std::forward<decltype(args)>(args)...); },
            std::move(args_tuple_local_copy));
        }
      });

  // Pack values and adjoints into new vars on main autodiff stack
  for (int i = 0; i < S; ++i) {
    result[i] = var(new precomputed_gradients_vari(
      values[i],
      nvars,
      &vari_map[i],
      &par_map[i]));
  }
}

/**
 * Evaluate a two-index loop in parallel
 */
template <typename ApplyFunction, typename IndexFunction,
          typename Res, typename... Args,
          require_st_var<Res>* = nullptr>
inline void parallel_map_impl(const ApplyFunction& app_fun,
                              const IndexFunction& index_fun,
                              Res&& result, int row_grainsize,
                              int col_grainsize, Args&&... x) {
    // Functors for manipulating vars at a given iteration of the loop
    auto var_counter = [&](auto&... xargs) {
      return count_vars(xargs...);
    };
    auto var_copier = [&](const auto&... xargs) {
      return std::tuple<decltype(deep_copy_vars(xargs))...>(
        deep_copy_vars(xargs)...);
    };
    auto vari_saver = [&](vari** varis) {
      return [=](const auto&... xargs) {
        save_varis(varis, xargs...);
      };
    };

    int R = result.rows();
    int C = result.cols();
    int S = result.size();

    // Assuming that the number of the vars at each iteration of the loop is
    // the same (as the operations at each iteration should be the same), we can
    // just count vars at the first iteration.
    int nvars = index_fun(0, 0, var_counter, x...);

    // Allocate memory for varis, values, and partials
    vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
      S * nvars);
    double* vals = ChainableStack::instance_->memalloc_.alloc_array<double>(
      S * nvars);
    double* partials = ChainableStack::instance_->memalloc_.alloc_array<double>(
      S * nvars);


    // By using a Map with an InnerStride & OuterStride we can index the Map
    // normally to get the location in memory for the varis/adjoints for that
    // iteration. Where the OuterStride (nvars*R) is the total length of each
    // column in memory, and the InnerStride (nvars) is the distance between
    // consecutive indexes.
    Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<-1, -1>> par_map(
      partials, R, C, Eigen::Stride<-1, -1>(nvars*R, nvars));
    Eigen::Map<stan::math::matrix_vi, 0, Eigen::Stride<-1, -1>> vari_map(
      varis, R, C, Eigen::Stride<-1, -1>(nvars*R, nvars));
    Eigen::Map<Eigen::MatrixXd> values(vals, R, C);

    tbb::parallel_for(
      tbb::blocked_range2d<size_t>(0, result.rows(), row_grainsize,
                                   0, result.cols(), col_grainsize),
      [&](
       const tbb::blocked_range2d<size_t>& r) {
        // Run nested autodiff in this scope
        const nested_rev_autodiff nested;

        for (size_t j = r.cols().begin(); j < r.cols().end(); ++j) {
          for (size_t i = r.rows().begin(); i < r.rows().end(); ++i) {
            index_fun(i, j, vari_saver(&vari_map(i, j)), x...);
            auto args_tuple_local_copy = index_fun(i, j, var_copier, x...);

            // Apply specified function to arguments at current iteration
            var out = apply([&](auto&&... args) { return app_fun(args...); },
                            args_tuple_local_copy);

            out.grad();

            // Extract value and adjoints to be put into vars on main
            // autodiff stack
            values(i, j) = std::move(out.vi_->val_);
            apply([&](auto&&... args) {
                save_adjoints(&par_map(i, j),
                              std::forward<decltype(args)>(args)...); },
              std::move(args_tuple_local_copy));
          }
        }
      });

  // Pack values and adjoints into new vars on main autodiff stack
  for (int j = 0; j < C; ++j) {
    for (int i = 0; i < R; ++i) {
      result(i, j) = var(new precomputed_gradients_vari(
        values(i, j),
        nvars,
        &vari_map(i, j),
        &par_map(i, j)));
    }
  }
}

}  // namespace math
}  // namespace stan
#endif
