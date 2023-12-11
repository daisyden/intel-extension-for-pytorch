#pragma once

#include <ATen/ATen.h>

#include <torch/csrc/jit/runtime/custom_operator.h>

#include "dil/dil.hpp"

namespace torch { namespace jit {

// XXX: PyTorch does not support nesting namespace
// And the alias analysis is not working for namespace other than aten ...
// So we fake some op namespaces to workaround that.
namespace ipex {
  static auto conv2d_relu = Symbol::fromQualString("ipex::conv2d_relu");
  static auto conv2d_sum = Symbol::fromQualString("ipex::conv2d_sum");
  // static auto conv2d_relu_sum = Symbol::fromQualString("ipex::conv2d_relu_sum");
  // static auto conv3d_relu_sum = Symbol::fromQualString("ipex::conv3d_relu_sum");
  static auto conv2d_sum_relu = Symbol::fromQualString("ipex::conv2d_sum_relu");
  static auto linear_relu = Symbol::fromQualString("ipex::linear_relu");
  static auto linear_gelu = Symbol::fromQualString("ipex::linear_gelu");

  // 3d ops
  static auto conv3d_relu = Symbol::fromQualString("ipex::conv3d_relu");
  static auto conv3d_sum = Symbol::fromQualString("ipex::conv3d_sum");
  static auto conv3d_sum_relu = Symbol::fromQualString("ipex::conv3d_sum_relu");

  static auto conv3d_instnorm_relu = Symbol::fromQualString("ipex::conv3d_instnorm_relu");
  static auto deconv3d = Symbol::fromQualString("ipex::deconv3d");
  static auto cat = Symbol::fromQualString("ipex::cat");
}

}} // namespace torch::jit

namespace torch_ipex {
namespace cpu {

class AtenIpexJITDev {
 public:
  // for JIT ops
  static at::Tensor dil_convolution_swish(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

  static at::Tensor dil_convolution_sigmoid(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

  static at::Tensor dil_convolution_clamp(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, float lower_bound, float upper_bound);

  static at::Tensor dil_convolution_relu(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);

  static at::Tensor dil_convolution_elu(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, float alpha, at::Scalar scale, at::Scalar input_scale);

  static at::Tensor& dil_convolution_sum(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor& accumu, at::Scalar alpha);

  static at::Tensor& dil_convolution_sum_relu( const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor& accumu, at::Scalar alpha);

  static at::Tensor dil_linear_fuse_eltwise(const at::Tensor& self, const at::Tensor& weight, const at::Tensor& bias, const dil::attr_t& attr);

  static at::Tensor dil_convolution_instancenorm_relu(const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, const at::Tensor& in_weight, const at::Tensor& in_bias);

  static at::Tensor dil_deconvolution3d(const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, int64_t groups, at::IntArrayRef dilation);

  static at::Tensor dil_concat3d(const at::TensorList tensors, const int64_t dim);

};

class Ipex_concat {
  public:
  Ipex_concat();

  dil::tensor dil_out_1;
  dil::tensor dil_out_2;
  dil::tensor dil_out_3;
  dil::tensor dil_out_4;
  dil::tensor dil_out_5;

};

}  // namespace cpu
}  // namespace torch_ipex
