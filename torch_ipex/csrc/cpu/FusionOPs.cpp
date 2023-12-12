#include "torch_ipex/csrc/cpu/FusionOPs.h"

#include <ATen/Context.h>
#include <ATen/InferSize.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <torch/csrc/autograd/function.h>

#include <limits>

#include "torch_ipex/csrc/cpu/int8/Config.h"
#include "torch_ipex/csrc/aten_ipex_bridge.h"
#include "torch_ipex/csrc/ipex_tensor_impl.h"
#include "torch_ipex/csrc/utils.h"
#include "torch_ipex/csrc/cpu/DevOPs.h"
#include "dbl/Common.h"
#include "dbl/Conv.h"
#include "dbl/Linear.h"
#include "dbl/Deconv.h"
#include "ShadeDataContext.h"

#include "dil/dil.hpp"
#include "cal_mean_var.hpp"

namespace torch_ipex {
namespace cpu {

using namespace dbl::comm;
using Time = std::chrono::high_resolution_clock;


at::Tensor dil_convolution_outplace_fusion(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const dil::attr_t& op_attr,
    const std::string& op_name = "Convolution_Relu") {
  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};
  // for int8 path, input always acbd format which is non-contiguous, .contiguous() will reorder to fp32
  auto src_dil_type = dbl::comm::try_gen_dil_tensor(input).get_data_type();
  auto input_contiguous = (src_dil_type == dil::data_type::u8 || src_dil_type == dil::data_type::s8
                           || IS_CONTIGUOUS_ANY(input)) ? input : input.contiguous();
  auto weight_dil_type = dbl::comm::try_gen_dil_tensor(weight).get_data_type();
  auto weight_contiguous = (weight_dil_type == dil::data_type::s8 || IS_CONTIGUOUS_ANY(weight)) ? weight : weight.contiguous();

  bool quantized = false;
  std::vector<float> output_scale = {};
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    int64_t num_ops_id = Int8OptConfig::fetch_and_add_ops_id();
    quantized = dbl::comm::get_int8_quantized_status(num_ops_id);
    std::vector<std::vector<float>> scales = dbl::comm::get_int8_scales(
        {input}, /*  uint8_used for output*/ false, num_ops_id);
    if (quantized) {
      output_scale.push_back(scales[1][0]);
      dbl::comm::reorder_to_int8_for_mix_prec(input_contiguous, scales[0]);
      dbl::comm::reorder_to_int8_for_mix_prec(weight_contiguous, {});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
      dbl::comm::reorder_to_dtype(weight, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input_contiguous);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight_contiguous);
  }

  dil_input = try_gen_dil_tensor(input_contiguous);
  dbl::conv::prepack_conv_weights(
    input_contiguous,
    dil_input,
    weight_contiguous,
    stride,
    padding,
    dilation,
    groups);
  dil_weight = try_gen_dil_tensor(weight_contiguous);

  if (bias.defined()) {
    auto bias_contiguous = IS_CONTIGUOUS_ANY(bias) ? bias : bias.contiguous();
    if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
      if (quantized) {
        auto src = dbl::comm::try_gen_dil_storage(bias_contiguous);
        auto src_type = src.get_data_type();
        if (src_type != dil::data_type::s32) {
          auto dst_desc = src.get_desc().to_type(dil::data_type::s32);
          auto bias_scales = dil_weight.get_scale();
          for (auto &scale : bias_scales) { scale *= dil_input.get_scale()[0];  }
          dbl::comm::reorder_to_desc(bias_contiguous, dst_desc, bias_scales);
        }
      } else {
        dbl::comm::reorder_to_dtype(bias_contiguous, at::kFloat);
      }
    } else {
      dbl::comm::reorder_to_bf16_for_mix_prec(bias_contiguous);
    }
    dil_bias = dbl::comm::try_gen_dil_tensor(bias_contiguous);
  }

  dil::tensor dil_output = dbl::conv::convolution_impl(
    dil_input,
    dil_weight,
    dil_bias,
    padding,
    stride,
    dilation,
    groups,
    op_attr,
    output_scale);

  auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(dil_output));
  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input_contiguous}, {aten_output}, op_name,
                              Int8OptConfig::fetch_and_add_ops_id());
  }
  return aten_output;
}

static at::Tensor& dil_convolution_inplace_fusion(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& accumu,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const dil::attr_t& attr,
    const std::string& op_name) {
  dil::tensor dil_input;
  dil::tensor dil_weight;
  dil::tensor dil_output;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  // for int8 path, input always acbd format which is non-contiguous, .contiguous() will reorder to fp32
  auto src_dil_type = dbl::comm::try_gen_dil_tensor(input).get_data_type();
  auto input_contiguous = (src_dil_type == dil::data_type::u8 || src_dil_type == dil::data_type::s8
                           || IS_CONTIGUOUS_ANY(input)) ? input : input.contiguous();
  auto weight_dil_type = dbl::comm::try_gen_dil_tensor(weight).get_data_type();
  auto weight_contiguous = (weight_dil_type == dil::data_type::s8 || IS_CONTIGUOUS_ANY(weight)) ? weight : weight.contiguous();
  auto ouput_dil_type = dbl::comm::try_gen_dil_tensor(accumu).get_data_type();
  auto output_contiguous = (ouput_dil_type == dil::data_type::u8 || ouput_dil_type == dil::data_type::s8 || IS_CONTIGUOUS_ANY(accumu)) ? accumu : accumu.contiguous();

  bool quantized = false;
  std::vector<float> output_scale = {};
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    int64_t num_ops_id = Int8OptConfig::fetch_and_add_ops_id();
    quantized = dbl::comm::get_int8_quantized_status(num_ops_id);
    std::vector<std::vector<float>> scales = dbl::comm::get_int8_scales(
        {input}, /*  uint8_used for output*/ false, num_ops_id);
    if (quantized) {
      output_scale.push_back(scales[1][0]);
      dbl::comm::reorder_to_int8_for_mix_prec(input_contiguous, scales[0]);
      dbl::comm::reorder_to_int8_for_mix_prec(weight_contiguous, {});
    } else {
      dbl::comm::reorder_to_dtype(input_contiguous, at::kFloat);
      dbl::comm::reorder_to_dtype(weight_contiguous, at::kFloat);
      // ouput may a int8 tensor, should reorder to fp32
      dbl::comm::reorder_to_dtype(output_contiguous, at::kFloat);
    }
  } else {
    dbl::comm::reorder_to_bf16_for_mix_prec(input_contiguous);
    dbl::comm::reorder_to_bf16_for_mix_prec(weight_contiguous);
    dbl::comm::reorder_to_bf16_for_mix_prec(output_contiguous);
  }

  dil_input = try_gen_dil_tensor(input_contiguous);
  dil_output = try_gen_dil_tensor(output_contiguous);

  dbl::conv::prepack_conv_weights(
    input_contiguous,
    dil_input,
    weight_contiguous,
    stride,
    padding,
    dilation,
    groups);
  dil_weight = try_gen_dil_tensor(weight_contiguous);

  if (bias.defined()) {
    auto bias_contiguous = IS_CONTIGUOUS_ANY(bias) ? bias : bias.contiguous();
    if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
      if (quantized) {
        auto src = dbl::comm::try_gen_dil_storage(bias_contiguous);
        auto src_type = src.get_data_type();
        if (src_type != dil::data_type::s32) {
          auto dst_desc = src.get_desc().to_type(dil::data_type::s32);
          auto bias_scales = dil_weight.get_scale();
          for (auto &scale : bias_scales) { scale *= dil_input.get_scale()[0];  }
          dbl::comm::reorder_to_desc(bias_contiguous, dst_desc, bias_scales);
        }
      } else {
        dbl::comm::reorder_to_dtype(bias_contiguous, at::kFloat);
      }
    } else {
      dbl::comm::reorder_to_bf16_for_mix_prec(bias_contiguous);
    }
    dil_bias = dbl::comm::try_gen_dil_tensor(bias_contiguous);
  }

  dbl::conv::convolution_inplace_impl(
    dil_input,
    dil_weight,
    dil_bias,
    dil_output,
    padding,
    stride,
    dilation,
    groups,
    attr,
    output_scale);

  dbl::comm::equip_dil_buffer(accumu, dil_output);
  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input_contiguous}, {accumu}, op_name,
                              Int8OptConfig::fetch_and_add_ops_id());
  }
  return accumu;
}

at::Tensor AtenIpexJITDev::dil_convolution_swish(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_swish", std::vector<c10::IValue>({}));
#endif
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_swish());
}

at::Tensor AtenIpexJITDev::dil_convolution_sigmoid(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_sigmoid", std::vector<c10::IValue>({}));
#endif
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_sigmoid());
}

at::Tensor AtenIpexJITDev::dil_convolution_clamp(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    float lower_bound,
    float upper_bound) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_clamp", std::vector<c10::IValue>({}));
#endif
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_clamp(lower_bound, upper_bound));
}

at::Tensor AtenIpexJITDev::dil_convolution_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_relu", std::vector<c10::IValue>({}));
#endif
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_relu(),
    "Convolution_Relu");
}

at::Tensor AtenIpexJITDev::dil_convolution_instancenorm_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    const at::Tensor& in_weight,
    const at::Tensor& in_bias) {

#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_instancenorm_relu", std::vector<c10::IValue>({}));
#endif

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    auto dil_input = try_gen_dil_tensor(input);
    auto dil_weight = try_gen_dil_tensor(weight);
    c10::optional<dil::tensor> dil_bias{c10::nullopt};
    dil::attr_t attr;
    dil::tensor dil_conv = dbl::conv::convolution_impl(dil_input, dil_weight, dil_bias, padding, stride, dilation, groups, attr, {});
    auto conv_output = dbl::comm::gen_aten_tensor_by(std::move(dil_conv));
    insert_or_updata_observer({input}, {conv_output}, "Conv3d", Int8OptConfig::fetch_and_add_ops_id());

    dil::tensor dil_norm = dbl::comm::try_gen_dil_tensor(conv_output);
    const dil::tensor norm_weight = dbl::comm::try_gen_dil_tensor(in_weight);
    const dil::tensor norm_bias = dbl::comm::try_gen_dil_tensor(in_bias);
    dil::tensor dil_norm_output;
    dil::batch_normalization_forward_inference::compute(
      dil_norm, norm_weight, norm_bias, dil_norm_output, 1e-5, {}, {}, dil::batch_normalization_flag::fuse_norm_relu);
    auto norm_output = dbl::comm::gen_aten_tensor_by(std::move(dil_norm_output));
    insert_or_updata_observer({conv_output}, {norm_output}, "BatchNorm", Int8OptConfig::fetch_and_add_ops_id());
    return norm_output;
  }

  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  bool quantized = true;
  std::vector<float> output_scale = {};
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    int64_t num_ops_id = Int8OptConfig::fetch_and_add_ops_id();
    // quantized = dbl::comm::get_int8_quantized_status(num_ops_id);
    std::vector<std::vector<float>> scales = dbl::comm::get_int8_scales({}, false, num_ops_id);
    if (quantized) {
      output_scale.push_back(scales[1][0]);
      dbl::comm::reorder_to_int8_for_mix_prec(input, scales[0]);
      dbl::comm::reorder_to_int8_for_mix_prec(weight, {});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
      dbl::comm::reorder_to_dtype(weight, at::kFloat);
    }
  }

  dil_input = try_gen_dil_tensor(input);
  dbl::conv::prepack_conv_weights(input, dil_input, weight, stride, padding, dilation, groups);
  dil_weight = try_gen_dil_tensor(weight);

  dil::attr_t attr;
  dil::tensor dil_conv = dbl::conv::convolution_impl(dil_input, dil_weight, dil_bias, padding, stride, dilation, groups, attr, output_scale);

  std::vector<float> bn_input_scales = {};
  std::vector<float> bn_output_scales = {};

  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    int64_t num_ops_id = Int8OptConfig::fetch_and_add_ops_id();
    bool quantized = dbl::comm::get_int8_quantized_status(num_ops_id);
    std::vector<std::vector<float>> scales = dbl::comm::get_int8_scales({}, false, num_ops_id);
    bn_input_scales = scales[0];
    bn_output_scales = scales[1];
  }

  const dil::tensor norm_weight = dbl::comm::try_gen_dil_tensor(in_weight);
  const dil::tensor norm_bias = dbl::comm::try_gen_dil_tensor(in_bias);
  dil::tensor dil_norm_output;
  double eps = 1e-5;

  auto src_type = dil_conv.get_data_type();
  if (src_type == dil::data_type::u8 || src_type == dil::data_type::s8) 
  {
    dil_norm_output = intel_mlperf::dil_instnorm_channel_last(dil_conv, in_weight, in_bias, bn_input_scales[0], bn_output_scales[0]);
    dil_norm_output.set_scale(bn_output_scales);
  } else
  {
    dil::batch_normalization_forward_inference::compute(
      dil_conv, norm_weight, norm_bias, dil_norm_output, eps, bn_input_scales, bn_output_scales, dil::batch_normalization_flag::fuse_norm_relu);
  }

  auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(dil_norm_output));
  return aten_output;
}

at::Tensor AtenIpexJITDev::dil_deconvolution3d(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_padding,
    int64_t groups,
    at::IntArrayRef dilation
    ) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_deconvolution3d", std::vector<c10::IValue>({}));
#endif

  auto weight_dil_type = dbl::comm::try_gen_dil_tensor(weight).is_public_format();
  if (weight_dil_type) { weight.transpose_(0, 1); }

  dil::tensor dil_input;
  dil::tensor dil_weight;
  c10::optional<dil::tensor> dil_bias{c10::nullopt};

  auto src_dil_type = dbl::comm::try_gen_dil_tensor(input).get_data_type();
  bool quantized = false;
  std::vector<float> output_scale = {};
  std::vector<float> input_scale = {};
  if (check_auto_mix_int8_fp32() && !check_int8_calibration()) {
    int64_t num_ops_id = Int8OptConfig::fetch_and_add_ops_id();
    quantized = dbl::comm::get_int8_quantized_status(num_ops_id);
    std::vector<std::vector<float>> scales = dbl::comm::get_int8_scales({}, false, num_ops_id);
    if (quantized) {
      output_scale.push_back(scales[1][0]);
      input_scale.push_back(scales[0][0]);
      dbl::comm::reorder_to_int8_for_mix_prec(input, scales[0]);
      dbl::comm::reorder_to_int8_for_mix_prec(weight, {});
    } else {
      dbl::comm::reorder_to_dtype(input, at::kFloat);
      dbl::comm::reorder_to_dtype(weight, at::kFloat);
    }
  }

  std::vector<int64_t> padding_r = {0, 0, 0};
  dil_input = dbl::comm::try_gen_dil_tensor(input);
  dbl::deconv::prepack_deconv3d_weights(input, weight, stride, padding, padding_r, output_padding, dilation, groups, false);
  dil_weight = dbl::comm::try_gen_dil_tensor(weight);

  dil::attr_t attr;
  if (quantized) {
    const dil::scale_t weights_scales;
    auto& weights_scales_in =
          dil_weight.has_scale() ? dil_weight.get_scale() : weights_scales;

    dil::scale_t bias_scales, op_scales;
    std::tie(bias_scales, op_scales) = dil::utils::compute_scales(
            input_scale[0], output_scale[0], weights_scales_in);
    attr.set_output_scales(2, op_scales);
  }

//#if defined(IPEX_PROFILE_OP)
//  RECORD_FUNCTION("dbl::deconv::deconvolution_impl", std::vector<c10::IValue>({}));
//#endif
  dil::tensor dil_output = dbl::deconv::deconvolution_impl(dil_input, dil_weight, dil_bias, padding, padding_r, output_padding, stride, dilation, groups, attr);
  auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(dil_output));

  if (check_auto_mix_int8_fp32() && check_int8_calibration()) {
    insert_or_updata_observer({input}, {aten_output}, "Deconv3d", Int8OptConfig::fetch_and_add_ops_id());
  }

  return aten_output;
}

Ipex_concat::Ipex_concat()
{
  dil::tensor::desc out_md_1 = dil::tensor::desc({1, 640, 8, 8, 8}, dil::tensor::data_type::s8, dil::tensor::format_tag::acdeb);
  dil::tensor::desc out_md_2 = dil::tensor::desc({1, 512, 16, 16, 16}, dil::tensor::data_type::s8, dil::tensor::format_tag::acdeb);
  dil::tensor::desc out_md_3 = dil::tensor::desc({1, 256, 32, 32, 32}, dil::tensor::data_type::s8, dil::tensor::format_tag::acdeb);
  dil::tensor::desc out_md_4 = dil::tensor::desc({1, 128, 64, 64, 64}, dil::tensor::data_type::s8, dil::tensor::format_tag::acdeb);
  dil::tensor::desc out_md_5 = dil::tensor::desc({1, 64, 128, 128, 128}, dil::tensor::data_type::s8, dil::tensor::format_tag::acdeb);
  dil_out_1.zero_init(out_md_1);
  dil_out_2.zero_init(out_md_2);
  dil_out_3.zero_init(out_md_3);
  dil_out_4.zero_init(out_md_4);
  dil_out_5.zero_init(out_md_5);
}

Ipex_concat ipex_concat;

at::Tensor AtenIpexJITDev::dil_concat3d(const at::TensorList tensors, const int64_t dim)
{
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_concat3d", std::vector<c10::IValue>({}));
#endif

  auto t1_ = dbl::comm::try_gen_dil_tensor(tensors[0]).get_data_type();
  if (t1_ == dil::data_type::f32)
  {
    return at::cat(tensors, dim);
  }
  else
  {
    auto dil_input1 = dbl::comm::try_gen_dil_tensor(tensors[0]);
    auto dil_input2 = dbl::comm::try_gen_dil_tensor(tensors[1]);

    auto in_sz = dil_input1.get_dims();
    dil::tensor dil_out;
    // if (in_sz[1] != 32)
    // {
    //   auto out_sz = {in_sz[0], in_sz[1] * 2, in_sz[2], in_sz[3], in_sz[4]};
    //   auto out_md = dil::tensor::desc(out_sz, dil::tensor::data_type::s8, dil::tensor::format_tag::acdeb);
    //   dil_out.init(out_md);
    // }
    // else
    // {
    //   dil_out = ipex_concat.dil_out_s;
    // }
    if (in_sz[1] == 32) {
      dil_out = ipex_concat.dil_out_5;
    } else if (in_sz[1] == 64) {
      dil_out = ipex_concat.dil_out_4;
    } else if (in_sz[1] == 128) {
      dil_out = ipex_concat.dil_out_3;
    } else if (in_sz[1] == 256) {
      dil_out = ipex_concat.dil_out_2;
    } else if (in_sz[1] == 320) {
      dil_out = ipex_concat.dil_out_1;
    }
    
    // auto start = Time::now();
    intel_mlperf::my_concat(dil_input1, dil_input2, dil_out, dim);
    // auto during = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - start).count();
    // printf(",,,concat,,,,,,,%f\n", (float)during * 1e-6);
    auto aten_output = dbl::comm::gen_aten_tensor_by(std::move(dil_out));
    return aten_output;
  }
}

at::Tensor AtenIpexJITDev::dil_convolution_elu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    float alpha,
    at::Scalar scale,
    at::Scalar input_scale) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_elu", std::vector<c10::IValue>({}));
#endif
  auto scale_value = scale.to<float>();
  auto input_scale_value = input_scale.to<float>();
  return dil_convolution_outplace_fusion(
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_elu(scale_value, alpha, input_scale_value));
}

at::Tensor& AtenIpexJITDev::dil_convolution_sum(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_sum", std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.to<float>();
  return dil_convolution_inplace_fusion(
    input,
    weight,
    bias,
    accumu,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::fuse_sum(scale),
    "Convolution_Sum");
}

at::Tensor& AtenIpexJITDev::dil_convolution_sum_relu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    at::Tensor& accumu,
    at::Scalar alpha) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_convolution_sum_relu", std::vector<c10::IValue>({}));
#endif
  auto scale = alpha.to<float>();
  at::Tensor& output = dil_convolution_inplace_fusion(
    input,
    weight,
    bias,
    accumu,
    stride,
    padding,
    dilation,
    groups,
    dil::attr_t::residual(scale),
    "Convolution_Sum_Relu");
  // if the next operator is convolution, u8 output can get a better performance than s8, so always convert
  // accumu's dil tensor to u8 data type.
  ShadeDataContext *shade_data_contex = (ShadeDataContext*)(output.storage().data_ptr().get_context());
  if (shade_data_contex->mix_prec_type == MIX_PREC_TYPE::MIX_INT8_FP32) {
    shade_data_contex->dil_tensor->to_type(dil::data_type::u8);
  }
  return output;
}

at::Tensor AtenIpexJITDev::dil_linear_fuse_eltwise(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const dil::attr_t& attr) {
#if defined(IPEX_PROFILE_OP)
  RECORD_FUNCTION("AtenIpexJITDev::dil_linear_fuse_eltwise", std::vector<c10::IValue>({}));
#endif
  return AtenIpexCPUDev::dil_linear(self, weight, bias, attr);
}

}  // namespace cpu
}  // namespace torch_ipex
