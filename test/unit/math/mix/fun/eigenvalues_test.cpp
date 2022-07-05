#include <test/unit/math/test_ad.hpp>
#include <stdexcept>
#include <stan/math/prim/fun/to_complex.hpp>

TEST(mathMixFun, eigenvalues) {
  using stan::math::to_complex;
  stan::test::ad_tolerances tols = stan::test::reverse_only_ad_tolerances();

  auto f = [](const auto& x) {
    using stan::math::eigenvalues;
    return eigenvalues(x);
  };
  for (const auto& x : stan::test::square_test_matrices(0, 2)) {
    stan::test::expect_ad(f, x);
    stan::test::expect_ad(tols, f, to_complex(x, 0).eval());
  }

  Eigen::MatrixXd a32(3, 2);
  a32 << 3, -5, 7, -7.2, 9.1, -6.3;
  EXPECT_THROW(f(a32), std::invalid_argument);
  EXPECT_THROW(f(to_complex(a32, 0)), std::invalid_argument);
}

// see eigenvectors_test.cpp for test of eigenvectors() and eigenvalues()
// using reconstruction identities
