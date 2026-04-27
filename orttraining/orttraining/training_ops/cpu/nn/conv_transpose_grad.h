#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class ConvTransposeGrad final : public OpKernel {
 public:
  explicit ConvTransposeGrad(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
    SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());
  }

  Status Compute(OpKernelContext* context) const override;

 protected:
  ConvAttributes conv_attrs_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvTransposeGrad);
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;
};

}  // namespace contrib
}  // namespace onnxruntime
