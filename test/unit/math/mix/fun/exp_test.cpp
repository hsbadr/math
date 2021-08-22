#include <test/unit/math/test_ad.hpp>

TEST(mathMixMatFun, exp) {
  auto f = [](const auto& x) {
    using stan::math::exp;
    return exp(x);
  };
  stan::test::expect_common_unary_vectorized<stan::test::PromoteToComplex::Yes>(
      f);
  stan::test::expect_unary_vectorized<stan::test::PromoteToComplex::Yes>(
      f, -15.2, -10, -0.5, 0.5, 1, 1.0, 1.3, 5, 10);
  stan::test::expect_complex_common(f);

  std::vector<double> com_args = stan::test::internal::common_nonzero_args();
  std::vector<double> args{-2.6, -0.5, 0.5, 1.5};

  stan::test::expect_ad_vector_matvar(f, stan::math::to_vector(com_args));
  stan::test::expect_ad_vector_matvar(f, stan::math::to_vector(args));
}

TEST(mathMixMatFun, exp_parmap_1d) {
  auto f = [](const auto& x) {
    // Functor defining how inputs should be indexed
    auto ind_f = [&](int i, const auto& fun, const auto& x) {
      return fun(x(i));
    };

    // Functor defining function to be applied to indexed arguments
    auto app_f = [&](const auto& x) { return stan::math::exp(x); };

    using rtn_type = stan::plain_type_t<decltype(x)>;

    rtn_type res(x.size());

    stan::math::parallel_map(app_f, ind_f, std::forward<rtn_type>(res), 1, x);

    return res;
  };

  Eigen::VectorXd in_vec = Eigen::VectorXd::Random(25);

  stan::test::expect_ad(f, in_vec);
}

TEST(mathMixMatFun, exp_parmap_2d) {
  auto f = [](const auto& x) {
    // Functor defining how inputs should be indexed
    auto ind_f = [&](int i, int j, const auto& fun, const auto& x) {
      return fun(x(i, j));
    };

    // Functor defining function to be applied to indexed arguments
    auto app_f = [&](const auto& x) { return stan::math::exp(x); };

    using rtn_type = stan::plain_type_t<decltype(x)>;

    rtn_type res(x.rows(), x.cols());

    stan::math::parallel_map(app_f, ind_f, std::forward<rtn_type>(res),
                             1, 1, x);

    return res;
  };

  Eigen::MatrixXd in_mat = Eigen::MatrixXd::Random(5, 5);

  stan::test::expect_ad(f, in_mat);
}
