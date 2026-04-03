// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime::contrib::test {

using namespace onnxruntime::test;

TEST(ResizeGradTest, ResizeGradBilinearWithSizes) {
  OpTester test("ResizeGrad", 1, kMSDomain);

  test.AddAttribute<std::string>("mode", "linear");
  test.AddAttribute<std::string>("coordinate_transformation_mode", "half_pixel");

  std::vector<float> dY(128, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 8, 8};

  std::vector<float> X(32, 1.0f);
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX(32, 4.0f);
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);

  test.AddOutput<float>("dX", dX_shape, dX);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr);
}

TEST(ResizeGradTest, ResizeGradBilinearWithSizesAndAlignCorners) {
  OpTester test("ResizeGrad", 1, kMSDomain);

  test.AddAttribute<std::string>("mode", "linear");
  test.AddAttribute<std::string>("coordinate_transformation_mode", "align_corners");

  std::vector<float> dY(128, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 8, 8};

  std::vector<float> X(32, 1.0f);
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX({2.9388f, 3.9184f, 3.9184f, 2.9388f, 3.9184f, 5.2245f, 5.2245f, 3.9184f,
                         3.9184f, 5.2245f, 5.2245f, 3.9184f, 2.9388f, 3.9184f, 3.9184f, 2.9388f,
                         2.9388f, 3.9184f, 3.9184f, 2.9388f, 3.9184f, 5.2245f, 5.2245f, 3.9184f,
                         3.9184f, 5.2245f, 5.2245f, 3.9184f, 2.9388f, 3.9184f, 3.9184f, 2.9388f});
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);

  test.AddOutput<float>("dX", dX_shape, dX);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr);
}

TEST(ResizeGradTest, ResizeGradBilinearWithScales) {
  OpTester test("ResizeGrad", 1, kMSDomain);

  test.AddAttribute<std::string>("mode", "linear");
  test.AddAttribute<std::string>("coordinate_transformation_mode", "half_pixel");

  std::vector<float> dY(72, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 6, 6};

  std::vector<float> X(32, 1.0f);
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX({2.7128f, 2.9550f, 2.7612f, 1.4533f, 2.9550f, 3.2189f, 3.0078f, 1.5830f,
                         2.7612f, 3.0078f, 2.8106f, 1.4792f, 1.4533f, 1.5830f, 1.4792f, 0.7785f,
                         2.7128f, 2.9550f, 2.7612f, 1.4533f, 2.9550f, 3.2189f, 3.0078f, 1.5830f,
                         2.7612f, 3.0078f, 2.8106f, 1.4792f, 1.4533f, 1.5830f, 1.4792f, 0.7785f});
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("", {0}, {});
  test.AddInput<float>("scales", {4}, {1.0f, 1.0f, 1.7f, 1.7f});

  test.AddOutput<float>("dX", dX_shape, dX);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr);
}

TEST(ResizeGradTest, ResizeGradBilinearWithScalesAndAlignCorners) {
  OpTester test("ResizeGrad", 1, kMSDomain);

  test.AddAttribute<std::string>("mode", "linear");
  test.AddAttribute<std::string>("coordinate_transformation_mode", "align_corners");

  std::vector<float> dY(72, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 6, 6};

  std::vector<float> X(32, 1.0f);
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX({1.9600f, 2.2400f, 2.2400f, 1.9600f, 2.2400f, 2.5600f, 2.5600f, 2.2400f,
                         2.2400f, 2.5600f, 2.5600f, 2.2400f, 1.9600f, 2.2400f, 2.2400f, 1.9600f,
                         1.9600f, 2.2400f, 2.2400f, 1.9600f, 2.2400f, 2.5600f, 2.5600f, 2.2400f,
                         2.2400f, 2.5600f, 2.5600f, 2.2400f, 1.9600f, 2.2400f, 2.2400f, 1.9600f});
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("", {0}, {});
  test.AddInput<float>("scales", {4}, {1.0f, 1.0f, 1.7f, 1.7f});

  test.AddOutput<float>("dX", dX_shape, dX);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr);
}

}  // namespace onnxruntime::contrib::test
