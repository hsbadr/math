#ifndef STAN_MATH_PRIM_FUN_POW_HPP
#define STAN_MATH_PRIM_FUN_POW_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/functor/apply_scalar_binary.hpp>
#include <cmath>
#include <complex>

namespace stan {
namespace math {

namespace internal {

/**
 * Return the first argument raised to the power of the second
 * argument.  At least one of the arguments must be a complex number.
 *
 * @tparam U type of base
 * @tparam V type of exponent
 * @param[in] x base
 * @param[in] y exponent
 * @return base raised to the power of the exponent
 */
template <typename U, typename V>
inline complex_return_t<U, V> complex_pow(const U& x, const V& y) {
  return exp(y * log(x));
}
}  // namespace internal

/**
 * Enables the vectorised application of the pow function, when
 * the first and/or second arguments are containers.
 *
 * @tparam T1 type of first input
 * @tparam T2 type of second input
 * @param a First input
 * @param b Second input
 * @return pow function applied to the two inputs.
 */
template <typename T1, typename T2, require_any_container_t<T1, T2>* = nullptr,
 require_all_not_matrix_t<T1, T2>* = nullptr>
inline auto pow(const T1& a, const T2& b) {
  return apply_scalar_binary(a, b, [&](const auto& c, const auto& d) {
    using std::pow;
    return pow(c, d);
  });
}

/**
 * Specialization for Matrices, scalars, and Eigen matrices of mixed scalar types.
 *
 * @tparam T1 A Scalar or type derived from EigenBase
 * @tparam T2 A Scalar or type derived from EigenBase
 * @param a base.
 * @param b The exponent to raise the base to.
 */
template <typename T1, typename T2, require_any_eigen_t<T1, T2>* = nullptr,
          require_all_not_st_var<T1, T2>* = nullptr,
          require_any_not_matrix_vt<std::is_arithmetic, T1, T2>* = nullptr>
inline auto pow(const T1& a, const T2& b) {
  return apply_scalar_binary(a, b, [&](const auto& c, const auto& d) {
    using std::pow;
    return pow(c, d);
  });
}

/**
 * Specialization for Arithmetic matrices
 * @tparam T1 Type derived from EigenBase with an Arithmetic scalar
 * @tparam T2 Type derived from EigenBase with an Arithmetic scalar
 * @param a base.
 * @param b The exponent to raise the base to.
 */
template <typename T1, typename T2, require_all_matrix_vt<std::is_arithmetic, T1, T2>* = nullptr>
inline auto pow(const T1& base, const T2& exponent) {
  return base.array().pow(exponent.array()).matrix();
}

}  // namespace math
}  // namespace stan
#endif
