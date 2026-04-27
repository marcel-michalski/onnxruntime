// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/nn/resize_grad.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status ResizeGrad<T>::Compute(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);

  const auto& dY_shape = dY->Shape();
  const auto& X_shape = X->Shape();

  ORT_ENFORCE(X_shape.NumDimensions() == 4, "Expected input tensor to have 4 dimensions. Actual: ",
              X_shape.NumDimensions());

  std::vector<int64_t> input_sizes(X_shape.GetDims().begin(), X_shape.GetDims().end());
  std::vector<int64_t> output_sizes(dY_shape.GetDims().begin(), dY_shape.GetDims().end());

  const float* scales_data = nullptr;
  if (context->InputCount() >= 4 && context->Input<Tensor>(3) != nullptr) {
    const Tensor* scales = context->Input<Tensor>(3);
    scales_data = scales->Data<float>();
  }

  std::vector<float> scales(4, 1.0f);
  if (scales_data != nullptr) {
    scales[0] = scales_data[0];
    scales[1] = scales_data[1];
    scales[2] = scales_data[2];
    scales[3] = scales_data[3];
  }

  Tensor* dX = context->Output(0, X_shape);
  std::fill(dX->MutableData<T>(), dX->MutableData<T>() + dX->Shape().Size(), static_cast<T>(0.0f));

  const T* dy_data = dY->Data<T>();
  T* dx_data = dX->MutableData<T>();

  if (mode_ == UpsampleMode::LINEAR) {
    return ComputeBilinearGrad(dy_data, dx_data, input_sizes, output_sizes, scales, X_shape.NumDimensions());
  }
  else {
    return ComputeNearestGrad(dy_data, dx_data, input_sizes, output_sizes, scales, X_shape.NumDimensions());
  }
}

template <typename T>
Status ResizeGrad<T>::ComputeBilinearGrad(const T* dY_data, T* dX_data,
                                         const std::vector<int64_t>& input_sizes,
                                         const std::vector<int64_t>& output_sizes,
                                         const std::vector<float>& scales,
                                         int64_t) const {
  const int64_t batch_size = input_sizes[0];
  const int64_t num_channels = input_sizes[1];
  const int64_t input_height = input_sizes[2];
  const int64_t input_width = input_sizes[3];
  const int64_t output_height = output_sizes[2];
  const int64_t output_width = output_sizes[3];

  if (input_height == output_height && input_width == output_width) {
    const size_t output_numel = batch_size * num_channels * output_height * output_width;
    std::memcpy(dX_data, dY_data, output_numel * sizeof(T));
    return Status::OK();
  }

  float rheight, rwidth;
  if (coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ALIGN_CORNERS) {
    if (output_height <= 1) {
      rheight = 0.0f;
    } else {
      rheight = static_cast<float>(input_height - 1) / static_cast<float>(output_height - 1);
    }
    if (output_width <= 1) {
      rwidth = 0.0f;
    } else {
      rwidth = static_cast<float>(input_width - 1) / static_cast<float>(output_width - 1);
    }
  } else {
    if (scales[2] != 1.0f) {
      rheight = 1.0f / scales[2];
    } else {
      rheight = static_cast<float>(input_height) / static_cast<float>(output_height);
    }
    if (scales[3] != 1.0f) {
      rwidth = 1.0f / scales[3];
    } else {
      rwidth = static_cast<float>(input_width) / static_cast<float>(output_width);
    }
  }

  const size_t dx_numel = batch_size * num_channels * input_height * input_width;
  std::fill(dX_data, dX_data + dx_numel, T(0.0f));

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; c++) {
      for (int64_t h2 = 0; h2 < output_height; ++h2) {
        for (int64_t w2 = 0; w2 < output_width; ++w2) {
          float h1r;
          if (coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ALIGN_CORNERS || coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ASYMMETRIC) {
            h1r = rheight * static_cast<float>(h2);
          } else {
            h1r = rheight * (static_cast<float>(h2) + 0.5f) - 0.5f;
            if (h1r < 0.0f) {
              h1r = 0.0f;
            }
          }

          float w1r;
          if (coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ALIGN_CORNERS || coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ASYMMETRIC) {
            w1r = rwidth * static_cast<float>(w2);
          } else {
            w1r = rwidth * (static_cast<float>(w2) + 0.5f) - 0.5f;
            if (w1r < 0.0f) {
              w1r = 0.0f;
            }
          }

          int h1 = static_cast<int>(h1r);
          int h1p = (h1 < input_height - 1) ? 1 : 0;
          float h1lambda = h1r - static_cast<float>(h1);
          float h0lambda = 1.0f - h1lambda;

          int w1 = static_cast<int>(w1r);
          int w1p = (w1 < input_width - 1) ? 1 : 0;
          float w1lambda = w1r - static_cast<float>(w1);
          float w0lambda = 1.0f - w1lambda;

          size_t dy_idx = ((n * num_channels + c) * output_height + h2) * output_width + w2;
          T d2val = dY_data[dy_idx];

          // Top-left
          size_t dx_idx = ((n * num_channels + c) * input_height + h1) * input_width + w1;
          dX_data[dx_idx] = T(static_cast<float>(dX_data[dx_idx]) + static_cast<float>(h0lambda * w0lambda) * static_cast<float>(d2val));

          // Top-right
          dx_idx = ((n * num_channels + c) * input_height + h1) * input_width + (w1 + w1p);
          dX_data[dx_idx] = T(static_cast<float>(dX_data[dx_idx]) + static_cast<float>(h0lambda * w1lambda) * static_cast<float>(d2val));

          // Bottom-left
          dx_idx = ((n * num_channels + c) * input_height + (h1 + h1p)) * input_width + w1;
          dX_data[dx_idx] = T(static_cast<float>(dX_data[dx_idx]) + static_cast<float>(h1lambda * w0lambda) * static_cast<float>(d2val));

          // Bottom-right
          dx_idx = ((n * num_channels + c) * input_height + (h1 + h1p)) * input_width + (w1 + w1p);
          dX_data[dx_idx] = T(static_cast<float>(dX_data[dx_idx]) + static_cast<float>(h1lambda * w1lambda) * static_cast<float>(d2val));
        }
      }
    }
  }

  return Status::OK();
}

template <typename T>
Status ResizeGrad<T>::ComputeNearestGrad(const T* dY_data, T* dX_data,
                                         const std::vector<int64_t>& input_sizes,
                                         const std::vector<int64_t>& output_sizes,
                                         const std::vector<float>& scales,
                                         int64_t) const {
  const int64_t batch_size = input_sizes[0];
  const int64_t num_channels = input_sizes[1];
  const int64_t input_height = input_sizes[2];
  const int64_t input_width = input_sizes[3];
  const int64_t output_height = output_sizes[2];
  const int64_t output_width = output_sizes[3];

  if (input_height == output_height && input_width == output_width) {
    const size_t output_numel = batch_size * num_channels * output_height * output_width;
    std::memcpy(dX_data, dY_data, output_numel * sizeof(T));
    return Status::OK();
  }

  const size_t dx_numel = batch_size * num_channels * input_height * input_width;
  std::fill(dX_data, dX_data + dx_numel, T(0.0f));

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t h2 = 0; h2 < output_height; ++h2) {
        for (int64_t w2 = 0; w2 < output_width; ++w2) {
          float h1r = get_original_coordinate_(static_cast<float>(h2), scales[2],
                                               static_cast<float>(output_height),
                                               static_cast<float>(input_height),
                                               0.0f, 0.0f);

          float w1r = get_original_coordinate_(static_cast<float>(w2), scales[3],
                                               static_cast<float>(output_width),
                                               static_cast<float>(input_width),
                                               0.0f, 0.0f);

          h1r = std::max(0.0f, std::min(h1r, static_cast<float>(input_height - 1)));
          w1r = std::max(0.0f, std::min(w1r, static_cast<float>(input_width - 1)));

          int64_t h1 = get_nearest_pixel_(h1r, output_height < input_height);
          int64_t w1 = get_nearest_pixel_(w1r, output_width < input_width);

          h1 = std::max<int64_t>(0, std::min<int64_t>(h1, input_height - 1));
          w1 = std::max<int64_t>(0, std::min<int64_t>(w1, input_width - 1));

          size_t dy_idx = ((n * num_channels + c) * output_height + h2) * output_width + w2;
          T d2val = dY_data[dy_idx];

          size_t dx_idx = ((n * num_channels + c) * input_height + h1) * input_width + w1;
          dX_data[dx_idx] = T(static_cast<float>(dX_data[dx_idx]) + static_cast<float>(d2val));
        }
      }
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    ResizeGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ResizeGrad<float>);

}  // namespace contrib
}  // namespace onnxruntime
