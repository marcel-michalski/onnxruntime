// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/upsamplebase.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class ResizeGrad final : public UpsampleBase, public OpKernel {
 public:
  ResizeGrad(const OpKernelInfo& info) : UpsampleBase(info), OpKernel(info) {
    ORT_ENFORCE(!antialias_, "Antialiasing is not supported in ResizeGrad yet.");

    ORT_ENFORCE(axes_.empty(), "ResizeGrad does not support the `axes` attribute yet.");

    std::string coordinate_transform_mode =
        info.GetAttrOrDefault<std::string>("coordinate_transformation_mode", "half_pixel");
    coordinate_transform_mode_ = StringToCoordinateTransformationMode(coordinate_transform_mode);
    ORT_ENFORCE(coordinate_transform_mode_ == ResizeCoordinateTransformationMode::HALF_PIXEL ||
                    coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ALIGN_CORNERS ||
                    coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ASYMMETRIC,
                "ResizeGrad only supports the `HALF_PIXEL`,  `ALIGN_CORNERS` and `ASYMMETRIC` coordinate_transform_mode ",
                coordinate_transform_mode, " is not supported yet.");

    ORT_ENFORCE(keep_aspect_ratio_policy_ == AspectRatioPolicy::STRETCH,
                "ResizeGrad only supports the `STRETCH` policy.");

    std::string mode;
    ORT_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());
    ORT_ENFORCE((UpsampleMode::LINEAR == mode_ || UpsampleMode::NN == mode_),
                "ResizeGrad only supports the `LINEAR` and `NN` mode. ", mode, " mode is not supported yet.");
  }

  Status Compute(OpKernelContext* context) const override;

  private:
   Status ComputeBilinearGrad(const T* dY_data, T* dX_data,
                             const std::vector<int64_t>& input_sizes,
                             const std::vector<int64_t>& output_sizes,
                             const std::vector<float>& scales,
                             int64_t rank) const;

   Status ComputeNearestGrad(const T* dY_data, T* dX_data,
                             const std::vector<int64_t>& input_sizes,
                             const std::vector<int64_t>& output_sizes,
                             const std::vector<float>& scales,
                             int64_t rank) const;

   ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ResizeGrad);
 };

}  // namespace contrib
}  // namespace onnxruntime
