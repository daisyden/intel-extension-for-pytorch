#pragma once
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "dil/dil.hpp"

namespace intel_mlperf {

    std::vector<at::Tensor> inst_mean_var(const at::Tensor& input);
    std::vector<at::Tensor> dil_inst_mean_var(const dil::tensor& input);
    std::vector<at::Tensor> dil_inst_mean_var_clast(const dil::tensor& input);

    at::Tensor instnorm_channel_last(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, float o_scale);
    dil::tensor dil_instnorm_channel_last(const dil::tensor& input, const at::Tensor& weight, const at::Tensor& bias, float i_scale, float o_scale);

    void my_concat(const dil::tensor& input1, const dil::tensor& input2, dil::tensor& output, int dim);

    template <int vec_length>
    class i_instancenorm_tpp {
        public:

        static void calc_mean_var(float *in, float *m, float *v, int64_t c, int64_t bl);
        static void calc_mean_var_int8(int8_t *in, float *m, float *v, int64_t c, int64_t bl);

        static void calc_norm(float *in, float *out, float *w, float *b, float *m, float *v, int64_t c, int64_t bl);
        static void calc_norm_int8(int8_t *in, int8_t *out, float *w, float *b, float *m, float *v, int64_t c, int64_t bl, float is, float os);
        static void calc_norm_int8_new(int8_t *in, int8_t *out, float *scale, float *shift, int64_t c, int64_t bl, float is, float os);
        
        static void ref(float *in, float &m, float &v, int64_t rl);
        static void ref(int8_t *in, float &m, float &v, int64_t rl);
        static void ref_concat(int8_t *in1, int8_t *in2, int8_t *out, int64_t c, int64_t bl);
        static void ref_concat32(int8_t *in1, int8_t *in2, int8_t *out, int64_t c, int64_t bl);
    };
}
