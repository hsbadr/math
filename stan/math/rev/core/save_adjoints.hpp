#ifndef STAN_MATH_REV_CORE_SAVE_ADJOINTS_HPP
#define STAN_MATH_REV_CORE_SAVE_ADJOINTS_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core/var.hpp>

#include <utility>
#include <vector>

namespace stan {
namespace math {

template <typename... Pargs>
inline double* save_adjoints(double* dest, const var& x, Pargs&&... args);

template <typename VarVec, require_std_vector_vt<is_var, VarVec>* = nullptr,
          typename... Pargs>
inline double* save_adjoints(double* dest, VarVec&& x, Pargs&&... args);

template <typename VecContainer,
          require_std_vector_st<is_var, VecContainer>* = nullptr,
          require_std_vector_vt<is_container, VecContainer>* = nullptr,
          typename... Pargs>
inline double* save_adjoints(double* dest, VecContainer&& x, Pargs&&... args);

template <typename EigT, require_eigen_vt<is_var, EigT>* = nullptr,
          typename... Pargs>
inline double* save_adjoints(double* dest, EigT&& x, Pargs&&... args);

template <typename Arith, require_arithmetic_t<scalar_type_t<Arith>>* = nullptr,
          typename... Pargs>
inline double* save_adjoints(double* dest, Arith&& x, Pargs&&... args);

inline double* save_adjoints(double* dest);

/**
 * Save the adjoint in x into the memory pointed to by dest,
 *   increment the dest storage pointer,
 *   recursively call save_adjoints on the rest of the arguments,
 *   and return the final value of the dest storage pointer.
 *
 * @tparam Pargs Types of remaining arguments
 * @param[in, out] dest Pointer to where adjoints are saved
 * @param[in] x A var
 * @param[in] args Additional arguments to have their adjoints saved
 * @return Final position of dest pointer
 */
template <typename... Pargs>
inline double* save_adjoints(double* dest, const var& x, Pargs&&... args) {
  *dest = x.vi_->adj_;
  return save_adjoints(dest + 1, std::forward<Pargs>(args)...);
}

/**
 * Save the adjoints in x into the memory pointed to by dest,
 *   increment the dest storage pointer,
 *   recursively call save_adjoints on the rest of the arguments,
 *   and return the final value of the dest storage pointer.
 *
 * @tparam VarVec A variant of std::vector<var>
 * @tparam Pargs Types of remaining arguments
 * @param[in, out] dest Pointer to where adjoints are saved
 * @param[in] x A std::vector of vars
 * @param[in] args Additional arguments to have their adjoints saved
 * @return Final position of dest pointer
 */
template <typename VarVec, require_std_vector_vt<is_var, VarVec>*,
          typename... Pargs>
inline double* save_adjoints(double* dest, VarVec&& x, Pargs&&... args) {
  for (int i = 0; i < x.size(); ++i) {
    dest[i] = x[i].vi_->adj_;
  }
  return save_adjoints(dest + x.size(), std::forward<Pargs>(args)...);
}

/**
 * Save the adjoints in x into the memory pointed to by dest,
 *   increment the dest storage pointer,
 *   recursively call save_adjoints on the rest of the arguments,
 *   and return the final value of the dest storage pointer.
 *
 * @tparam VecContainer std::vector<T> where T is another type containing vars
 * @tparam Pargs Types of remaining arguments
 * @param[in, out] dest Pointer to where adjoints are saved
 * @param[in] x A std::vector of containers containing of vars
 * @param[in] args Additional arguments to have their adjoints saved
 * @return Final position of dest pointer
 */
template <typename VecContainer, require_std_vector_st<is_var, VecContainer>*,
          require_std_vector_vt<is_container, VecContainer>*, typename... Pargs>
inline double* save_adjoints(double* dest, VecContainer&& x, Pargs&&... args) {
  for (size_t i = 0; i < x.size(); ++i) {
    dest = save_adjoints(dest, x[i]);
  }
  return save_adjoints(dest, std::forward<Pargs>(args)...);
}

/**
 * Save the adjoints in x into the memory pointed to by dest,
 *   increment the dest storage pointer,
 *   recursively call save_adjoints on the rest of the arguments,
 *   and return the final value of the dest storage pointer.
 *
 * @tparam EigT An Eigen type with var value type
 * @tparam Pargs Types of remaining arguments
 * @param[in, out] dest Pointer to where adjoints are saved
 * @param[in] x An Eigen container of vars
 * @param[in] args Additional arguments to have their adjoints saved
 * @return Final position of dest pointer
 */
template <typename EigT, require_eigen_vt<is_var, EigT>*, typename... Pargs>
inline double* save_adjoints(double* dest, EigT&& x, Pargs&&... args) {
  for (int i = 0; i < x.size(); ++i) {
    dest[i] = x(i).vi_->adj_;
  }
  return save_adjoints(dest + x.size(), std::forward<Pargs>(args)...);
}

/**
 * Ignore arithmetic types.
 *
 * Recursively call save_adjoints on the rest of the arguments
 *   and return the final value of the dest storage pointer.
 *
 * @tparam Arith An arithmetic type
 * @tparam Pargs Types of remaining arguments
 * @param[in, out] dest Pointer to where adjoints are saved
 * @param[in] x An argument not containing vars
 * @param[in] args Additional arguments to have their adjoints saved
 * @return Final position of dest pointer
 */
template <typename Arith, require_arithmetic_t<scalar_type_t<Arith>>*,
          typename... Pargs>
inline double* save_adjoints(double* dest, Arith&& x, Pargs&&... args) {
  return save_adjoints(dest, std::forward<Pargs>(args)...);
}

/**
 * End save_adjoints recursion and return pointer
 *
 * @param dest Pointer
 */
inline double* save_adjoints(double* dest) { return dest; }

}  // namespace math
}  // namespace stan

#endif
