#include <stan/math.hpp>
#include <test/unit/util.hpp>
#include <iostream>
#include <gtest/gtest.h>

TEST(MathFunctions, parall_map) {
  using stan::math::pow;
  using stan::math::var;
  using stan::math::vector_v;
  vector_v in1_par = vector_v::Random(10);
  vector_v in2_par = vector_v::Random(10);
  vector_v in1_ser = in1_par;
  vector_v in2_ser = in2_par;
  vector_v out_par(10);
  vector_v out_ser(10);

  for (int i = 0; i < 10; ++i) {
    out_ser[i] = in1_ser[i] * 0.5 + exp(in2_ser[i]);
  }

  // Functor defining how inputs should be indexed
  auto ind_f = [&](int i, const auto& fun,
                    const auto& x, const auto& y, const auto& z) {
    return fun(x(i), y, z(i));
  };

  // Functor defining function to be applied to indexed arguments
  auto f = [&](const auto& x, const auto& y, const auto& z) {
    return x * y + exp(z); };

  parallel_map(f, ind_f, std::forward<vector_v>(out_par), 1,
                         in1_par, 0.5, in2_par);
  EXPECT_MATRIX_EQ(out_par.val(), out_ser.val());
  EXPECT_MATRIX_EQ(out_par.adj(), out_ser.adj());

  for (int i = 0; i < 10; ++i) {
    out_par[i].grad();
    out_ser[i].grad();
  }

  EXPECT_MATRIX_EQ(in1_par.adj(), in1_ser.adj());
  EXPECT_MATRIX_EQ(in2_par.adj(), in2_ser.adj());

  stan::math::recover_memory();
}

TEST(MathFunctions, parall_map_vec) {
  using stan::math::pow;
  using stan::math::var;
  using stan::math::vector_v;
  vector_v in1_par = vector_v::Random(10);
  vector_v in2_par = vector_v::Random(10);
  vector_v in3_par = vector_v::Random(10);
  vector_v in1_ser = in1_par;
  vector_v in2_ser = in2_par;
  vector_v in3_ser = in3_par;
  vector_v out_par(10);
  vector_v out_ser(10);

  for (int i = 0; i < 10; ++i) {
    out_ser[i] = in1_ser[i] * sum(in2_ser) + exp(in3_ser[i]);
  }

  // Functor defining how inputs should be indexed
  auto ind_f = [&](int i, const auto& fun,
                    const auto& x, const auto& y, const auto& z) {
    return fun(x(i), y, z(i));
  };

  // Functor defining function to be applied to indexed arguments
  auto f = [&](const auto& x, const auto& y, const auto& z) {
    return x * sum(y) + exp(z); };

  parallel_map(f, ind_f, std::forward<vector_v>(out_par), 1,
                         in1_par, in2_par, in3_par);
  EXPECT_MATRIX_EQ(out_par.val(), out_ser.val());
  EXPECT_MATRIX_EQ(out_par.adj(), out_ser.adj());

  for (int i = 0; i < 10; ++i) {
    out_par[i].grad();
    out_ser[i].grad();
  }

  EXPECT_MATRIX_EQ(in1_par.adj(), in1_ser.adj());
  EXPECT_MATRIX_EQ(in2_par.adj(), in2_ser.adj());
  EXPECT_MATRIX_EQ(in3_par.adj(), in3_ser.adj());

  stan::math::recover_memory();
}


TEST(MathFunctions, parall_map_var_2d) {
  using stan::math::pow;
  using stan::math::var;
  using stan::math::matrix_v;
  matrix_v in1_par = matrix_v::Random(10, 10);
  matrix_v in2_par = matrix_v::Random(10, 10);
  matrix_v in1_ser = in1_par;
  matrix_v in2_ser = in2_par;
  matrix_v out_par(10, 10);
  matrix_v out_ser(10, 10);

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      out_ser(i, j) = in1_ser(i, j) * 0.5 + exp(in2_ser(i, j));
    }
  }

  // Functor defining how inputs should be indexed
  auto ind_f = [&](int i, int j, const auto& fun,
                    const auto& x, const auto& y, const auto& z) {
    return fun(x(i, j), y, z(i, j));
  };

  // Functor defining function to be applied to indexed arguments
  auto f = [&](const auto& x, const auto& y, const auto& z) {
    return x * y + exp(z); };

  parallel_map(f, ind_f, std::forward<stan::math::matrix_v>(out_par), 1, 1,
                         in1_par, 0.5, in2_par);

  EXPECT_MATRIX_EQ(out_par.val(), out_ser.val());
  EXPECT_MATRIX_EQ(out_par.adj(), out_ser.adj());

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j< 10; ++j) {
      out_ser(i, j).grad();
      out_par(i, j).grad();
    }
  }

  EXPECT_MATRIX_EQ(in1_par.adj(), in1_ser.adj());
  EXPECT_MATRIX_EQ(in2_par.adj(), in2_ser.adj());

  stan::math::recover_memory();
}
