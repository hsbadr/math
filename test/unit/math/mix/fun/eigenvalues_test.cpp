#include <test/unit/math/test_ad.hpp>

TEST(mathMixFun, eigenvalues) {
  auto f = [](const auto& x) {
    using stan::math::eigenvalues;
    return eigenvalues(x);
  };
  for (const auto& x : stan::test::square_test_matrices(0, 2)) {
    stan::test::expect_ad(f, x);
  }
}

// see eigenvectors_test.cpp for test of eigenvectors() and eigenvalues()
// using reconstruction identities
