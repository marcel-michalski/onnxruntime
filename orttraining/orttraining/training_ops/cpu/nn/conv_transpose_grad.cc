// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/nn/conv_transpose_grad.h"
#include "core/common/narrow.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status ConvTransposeGrad<T>::Compute(OpKernelContext* context) const {
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X  = context->Input<Tensor>(1);
  const Tensor* W  = context->Input<Tensor>(2);

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[1] * conv_attrs_.group;

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));
  const size_t kernel_rank = kernel_shape.size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();

  ConvAttributes::ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_rank * 2, 0);
  }

  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_rank, 1);
  }

  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_rank, 1);
  }

  TensorShape input_shape = X->Shape().Slice(2);
  TensorShape output_shape = dY->Shape().Slice(2);

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();

  const int64_t C_per_group = C / conv_attrs_.group;
  const int64_t M_per_group = M / conv_attrs_.group;

  const int64_t X_offset  = C_per_group * input_image_size;
  const int64_t dY_offset = M_per_group * output_image_size;
  const int64_t W_offset  = W->Shape().Size() / conv_attrs_.group;

  const int64_t kernel_dim = M_per_group * kernel_size;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  const size_t col_buffer_size = narrow<size_t>(kernel_dim * input_image_size * sizeof(T));
  BufferUniquePtr col_buffer(alloc->Alloc(col_buffer_size), BufferDeleter(alloc));
  T* col_buffer_data = static_cast<T*>(col_buffer.get());

  const size_t dW_temp_size = narrow<size_t>(W_offset * sizeof(T));
  BufferUniquePtr dW_temp(alloc->Alloc(dW_temp_size), BufferDeleter(alloc));
  T* dW_temp_data = static_cast<T*>(dW_temp.get());

  BufferUniquePtr bias_multiplier(alloc->Alloc(narrow<size_t>(output_image_size * sizeof(T))),
                                  BufferDeleter(alloc));
  T* bias_multiplier_data = static_cast<T*>(bias_multiplier.get());
  math::Set<T, CPUMathUtil>(narrow<ptrdiff_t>(output_image_size), static_cast<T>(1),
                            bias_multiplier_data, &CPUMathUtil::Instance());

  Tensor* dB = context->Output(2, {M});
  T* dBdata = nullptr;
  if (dB) {
    dBdata = dB->template MutableData<T>();
    math::Set<T, CPUMathUtil>(narrow<ptrdiff_t>(M), static_cast<T>(0), dBdata, &CPUMathUtil::Instance());
  }

  Tensor* dW = context->Output(1, W->Shape());
  T* dWdata = nullptr;
  if (dW) {
    dWdata = dW->template MutableData<T>();
    math::Set<T, CPUMathUtil>(narrow<ptrdiff_t>(W->Shape().Size()), static_cast<T>(0),
                              dWdata, &CPUMathUtil::Instance());
  }

  Tensor* dX = context->Output(0, X->Shape());
  T* dXdata = nullptr;
  if (dX) {
    dXdata = dX->template MutableData<T>();
    math::Set<T, CPUMathUtil>(narrow<ptrdiff_t>(X->Shape().Size()), static_cast<T>(0),
                              dXdata, &CPUMathUtil::Instance());
  }

  const T* Xdata   = X->template Data<T>();
  const T* Wdata   = W->template Data<T>();
  const T* dYdata  = dY->template Data<T>();

  bool skip_im2col = (kernel_size == 1) && conv_attrs_.HasStridesOneAndNoPadding();

  auto output_dims = output_shape.AsShapeVector();
  auto input_dims  = input_shape.AsShapeVector();

  for (int image_id = 0; image_id < N; ++image_id) {
    if (dW) {
      for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
        if (!skip_im2col) {
          math::Im2col<T, StorageOrder::NCHW>()(
              dYdata + group_id * dY_offset,
              output_dims.data(),
              input_dims.data(),
              kernel_dim,
              kernel_shape.data(),
              strides.data(),
              dilations.data(),
              pads.data(),
              static_cast<int>(kernel_rank),
              col_buffer_data);
        }

        math::Gemm<T>(
            CblasNoTrans, CblasTrans,
            narrow<ptrdiff_t>(kernel_dim),
            narrow<ptrdiff_t>(C_per_group),
            narrow<ptrdiff_t>(input_image_size),
            1.0f,
            skip_im2col ? dYdata + group_id * dY_offset : col_buffer_data,
            Xdata + group_id * X_offset,
            0.0f,
            dW_temp_data,
            tp,
            &mlas_backend_kernel_selector_config_);

        T* group_dW = dWdata + group_id * W_offset;
        for (int64_t m = 0; m < M_per_group; ++m) {
          for (int64_t k = 0; k < kernel_size; ++k) {
            for (int64_t c = 0; c < C_per_group; ++c) {
              int64_t src_idx = (m * kernel_size + k) * C_per_group + c;
              int64_t dst_idx = c * M_per_group * kernel_size + m * kernel_size + k;
              group_dW[dst_idx] += dW_temp_data[src_idx];
            }
          }
        }
      }
    }

    if (dB) {
      math::Gemv<T, CPUMathUtil>(
          CblasNoTrans,
          static_cast<int>(M),
          static_cast<int>(output_image_size),
          1.0f,
          dYdata,
          bias_multiplier_data,
          1.0f,
          dBdata,
          &CPUMathUtil::Instance());
    }

    Xdata  += X_offset * conv_attrs_.group;
    dYdata += dY_offset * conv_attrs_.group;
  }

  if (dX) {
    dYdata = dY->template Data<T>();
    Wdata  = W->template Data<T>();

    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
        if (!skip_im2col) {
          math::Im2col<T, StorageOrder::NCHW>()(
              dYdata + group_id * dY_offset,
              output_dims.data(),
              input_dims.data(),
              kernel_dim,
              kernel_shape.data(),
              strides.data(),
              dilations.data(),
              pads.data(),
              static_cast<int>(kernel_rank),
              col_buffer_data);
        }

        math::Gemm<T>(
            CblasNoTrans, CblasNoTrans,
            narrow<ptrdiff_t>(C_per_group),
            narrow<ptrdiff_t>(input_image_size),
            narrow<ptrdiff_t>(kernel_dim),
            1.0f,
            Wdata + group_id * W_offset,
            skip_im2col ? dYdata + group_id * dY_offset : col_buffer_data,
            1.0f,
            dXdata + group_id * X_offset,
            tp,
            &mlas_backend_kernel_selector_config_);
      }
      dXdata += X_offset * conv_attrs_.group;
      dYdata += dY_offset * conv_attrs_.group;
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    ConvTransposeGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvTransposeGrad<float>);

}  // namespace contrib
}  // namespace onnxruntime
