#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include "cal_mean_var.hpp"
#include "el_common_intrin.hpp"

namespace intel_mlperf {

template <>
void i_instancenorm_tpp<16>::calc_mean_var(float *in, float *m, float *v, int64_t c, int64_t bl) 
{
  auto vlen = c / 16;
  __m512 sm[vlen];
  __m512 smm[vlen];
  for (int i = 0; i < vlen; ++i)
  {
    sm[i] = _mm512_setzero_ps();
    smm[i] = _mm512_setzero_ps();
  }

  auto* pin = in;
  auto rbl =_mm512_set1_ps(1.0 / bl);
  for (auto i = 0; i < bl; ++i)
  {
    for (auto j = 0; j < vlen; ++j)
    {
      auto f = _mm512_loadu_ps(&pin[(i * vlen + j) * 16]);
      sm[j] += f;
      smm[j] += f * f;
    }
  }

  for (auto i = 0; i < vlen; ++i)
  {
    _mm512_storeu_ps(&m[i * 16], sm[i] * rbl);
    _mm512_storeu_ps(&v[i * 16], smm[i] * rbl);
  }
}

template <>
void i_instancenorm_tpp<16>::calc_mean_var_int8(int8_t *in, float *m, float *v, int64_t c, int64_t bl) 
{
  auto vlen = c / 16;
  __m512 sm[vlen];
  __m512 smm[vlen];
  for (int i = 0; i < vlen; ++i)
  {
    sm[i] = _mm512_setzero_ps();
    smm[i] = _mm512_setzero_ps();
  }

  auto* pin = in;
  auto rbl =_mm512_set1_ps(1.0 / bl);
  for (auto i = 0; i < bl; ++i)
  {
    for (auto j = 0; j < vlen; ++j)
    {
      auto f = _mm512_loadu_i8_to_fp32(&pin[(i * vlen + j) * 16]);
      sm[j] += f;
      smm[j] += f * f;
    }
  }

  for (auto i = 0; i < vlen; ++i)
  {
    _mm512_storeu_ps(&m[i * 16], sm[i] * rbl);
    _mm512_storeu_ps(&v[i * 16], smm[i] * rbl);
  }
}

template <>
void i_instancenorm_tpp<16>::calc_norm(float *in, float *out, float *w, float *b, float *m, float *v, int64_t c, int64_t bl) 
{
  auto vlen = c / 16;
  __m512 _m[vlen];
  __m512 _v[vlen];
  __m512 _w[vlen];
  __m512 _b[vlen];
  for (int i = 0; i < vlen; ++i)
  {
    _m[i] = _mm512_setzero_ps();
    _v[i] = _mm512_setzero_ps();
    _w[i] = _mm512_setzero_ps();
    _b[i] = _mm512_setzero_ps();
  }

  auto veps = _mm512_set1_ps(1e-5);
  for (auto i = 0; i < vlen; ++i)
  {
    _m[i] = _mm512_loadu_ps(&m[i * 16]);
    _v[i] = 1. / _mm512_sqrt_ps(_mm512_loadu_ps(&v[i * 16]) + veps);
    _w[i] = _mm512_loadu_ps(&w[i * 16]);
    _b[i] = _mm512_loadu_ps(&b[i * 16]);
  }

  auto* pin = in;
  auto* pout = out;
  for (auto i = 0; i < bl; ++i)
  {
    for (auto j = 0; j < vlen; ++j)
    {
      auto f = _mm512_loadu_ps(&pin[(i * vlen + j) * 16]);
      auto o = (f - _m[j]) * _w[j] * _v[j] + _b[j];
      _mm512_storeu_ps(&pout[(i * vlen + j) * 16], o);
    }
  }
}

template <>
void i_instancenorm_tpp<16>::calc_norm_int8(int8_t *in, int8_t *out, float *w, float *b, float *m, float *v, int64_t c, int64_t bl, float is, float os) 
{
  auto vlen = c / 16;
  __m512 _m[vlen];
  __m512 _v[vlen];
  __m512 _w[vlen];
  __m512 _b[vlen];
  for (int i = 0; i < vlen; ++i)
  {
    _m[i] = _mm512_setzero_ps();
    _v[i] = _mm512_setzero_ps();
    _w[i] = _mm512_setzero_ps();
    _b[i] = _mm512_setzero_ps();
  }

  auto veps = _mm512_set1_ps(1e-5);
  for (auto i = 0; i < vlen; ++i)
  {
    _m[i] = _mm512_loadu_ps(&m[i * 16]);
    _v[i] = 1. / _mm512_sqrt_ps(_mm512_loadu_ps(&v[i * 16]) + veps);
    _w[i] = _mm512_loadu_ps(&w[i * 16]);
    _b[i] = _mm512_loadu_ps(&b[i * 16]);
  }

  auto* pin = in;
  auto* pout = out;
  auto voscale = _mm512_set1_ps(os);
  auto viscale = _mm512_set1_ps(is);
  auto vo_off = _mm_set1_epi8(0);

  for (auto i = 0; i < bl; ++i)
  {
    for (auto j = 0; j < vlen; ++j)
    {
      auto f = _mm512_loadu_i8_to_fp32(&pin[(i * vlen + j) * 16]);
      auto o = (f - _m[j]) * _w[j] * _v[j] + _b[j];
      auto r = _mm512_scale_minmax_i8_ps(o, voscale);
      _mm512_mask_cvtepi32_storeu_epi8(&pout[(i * vlen + j) * 16], 0xffff, r, vo_off);
    }
  }
}

template <>
void i_instancenorm_tpp<16>::calc_norm_int8_new(int8_t *in, int8_t *out, float *sc, float *sh, int64_t c, int64_t bl, float is, float os) 
{
  auto vlen = c / 16;
  __m512 vscale[vlen];
  __m512 vshift[vlen];
  for (int i = 0; i < vlen; ++i)
  {
    vscale[i] = _mm512_setzero_ps();
    vshift[i] = _mm512_setzero_ps();
  }

  auto voscale = _mm512_set1_ps(os);
  auto viscale = _mm512_set1_ps(is);
  auto veps = _mm512_set1_ps(1e-5);

  for (auto i = 0; i < vlen; ++i)
  {
    vscale[i] = _mm512_loadu_ps(&sc[i * 16]);
    vshift[i] = _mm512_loadu_ps(&sh[i * 16]);
  }

  auto* pin = in;
  auto* pout = out;
  auto vo_off = _mm_set1_epi8(0);

  for (auto i = 0; i < bl; ++i)
  {
    for (auto j = 0; j < vlen; ++j)
    {
      auto f = _mm512_loadu_i8_to_fp32(&pin[(i * vlen + j) * 16]);
      auto o = _mm512_fmadd_ps(f, vscale[j], vshift[j]);
      auto r = _mm512_scale_minmax_i8_ps(o, voscale);
      _mm512_mask_cvtepi32_storeu_epi8(&pout[(i * vlen + j) * 16], 0xffff, r, vo_off);
    }
  }
}

at::Tensor instnorm_channel_last(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, float o_scale) 
{
  auto in_sz = input.sizes();
  auto batch = in_sz[0];
  auto channel = in_sz[1];
  auto block_len = in_sz[2];
  auto reduce_l = in_sz[2] * in_sz[3] * in_sz[4];
  auto block_num = reduce_l / block_len;
  auto mean_t = at::empty({batch, block_num, channel}, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto var_t = at::empty({batch, block_num, channel}, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto output = at::empty(in_sz, input.options().memory_format(input.suggest_memory_format()));

  auto* in_ptr = input.data_ptr();
  auto* out_ptr = output.data_ptr();
  auto* w_ptr = weight.data_ptr();
  auto* b_ptr = bias.data_ptr();
  auto* m_ptr = mean_t.data_ptr();
  auto* v_ptr = var_t.data_ptr();
  auto data_type = input.scalar_type();

  // calculate mean and variance
  if (data_type == c10::ScalarType::Float) {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<float (*)[block_len * channel]>(in_ptr);
      auto* m = reinterpret_cast<float (*)[channel]>(m_ptr);
      auto* v = reinterpret_cast<float (*)[channel]>(v_ptr);
      i_instancenorm_tpp<16>::calc_mean_var(bin[i], m[i], v[i], channel, block_len);
    }
  } else if (data_type == c10::ScalarType::Char)
  {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<int8_t (*)[block_len * channel]>(in_ptr);
      auto* m = reinterpret_cast<float (*)[channel]>(m_ptr);
      auto* v = reinterpret_cast<float (*)[channel]>(v_ptr);
      i_instancenorm_tpp<16>::calc_mean_var_int8(bin[i], m[i], v[i], channel, block_len);
    }
  }

  auto mt = at::mean(mean_t, 1);
  auto mt2 = at::mean(var_t, 1);
  auto vt = mt2 - mt * mt;
  auto* mt_ptr = mt.data_ptr();
  auto* vt_ptr = vt.data_ptr();

  // calculate norm
  if (data_type == c10::ScalarType::Float) {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<float (*)[block_len * channel]>(in_ptr);
      auto* w = reinterpret_cast<float (*)>(w_ptr);
      auto* b = reinterpret_cast<float (*)>(b_ptr);
      auto* m = reinterpret_cast<float (*)[channel]>(mt_ptr);
      auto* v = reinterpret_cast<float (*)[channel]>(vt_ptr);
      auto* bout = reinterpret_cast<float (*)[block_len * channel]>(out_ptr);
      i_instancenorm_tpp<16>::calc_norm(bin[i], bout[i], w, b, m[i/block_num], v[i/block_num], channel, block_len);
    }
  } else if (data_type == c10::ScalarType::Char) {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<int8_t (*)[block_len * channel]>(in_ptr);
      auto* w = reinterpret_cast<float (*)>(w_ptr);
      auto* b = reinterpret_cast<float (*)>(b_ptr);
      auto* m = reinterpret_cast<float (*)[channel]>(mt_ptr);
      auto* v = reinterpret_cast<float (*)[channel]>(vt_ptr);
      auto* bout = reinterpret_cast<int8_t (*)[block_len * channel]>(out_ptr);
      i_instancenorm_tpp<16>::calc_norm_int8(bin[i], bout[i], w, b, m[i/block_num], v[i/block_num], channel, block_len, 1.0, o_scale);
    }  
  }
  return output;
}

dil::tensor dil_instnorm_channel_last(const dil::tensor& input, const at::Tensor& weight, const at::Tensor& bias, float i_scale, float o_scale) 
{
#if defined(IPEX_PROFILE_OP)
RECORD_FUNCTION("dil_instnorm_channel_last", std::vector<c10::IValue>({}));
#endif

  auto in_sz = input.get_dims();
  auto batch = in_sz[0];
  auto channel = in_sz[1];
  auto block_len = in_sz[2];
  auto reduce_l = in_sz[2] * in_sz[3] * in_sz[4];
  auto block_num = reduce_l / block_len;
  auto mean_t = at::empty({batch, block_num, channel}, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto var_t = at::empty({batch, block_num, channel}, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto* in_ptr = input.get_data_handle();
  auto* w_ptr = weight.data_ptr();
  auto* b_ptr = bias.data_ptr();
  auto* m_ptr = mean_t.data_ptr();
  auto* v_ptr = var_t.data_ptr();
  auto data_type = input.get_data_type();

  // calculate mean and variance
  if (data_type == dil::data_type::f32) {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<float (*)[block_len * channel]>(in_ptr);
      auto* m = reinterpret_cast<float (*)[channel]>(m_ptr);
      auto* v = reinterpret_cast<float (*)[channel]>(v_ptr);
      i_instancenorm_tpp<16>::calc_mean_var(bin[i], m[i], v[i], channel, block_len);
    }
  } else if (data_type == dil::data_type::u8 || data_type == dil::data_type::s8)
  {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<int8_t (*)[block_len * channel]>(in_ptr);
      auto* m = reinterpret_cast<float (*)[channel]>(m_ptr);
      auto* v = reinterpret_cast<float (*)[channel]>(v_ptr);
      i_instancenorm_tpp<16>::calc_mean_var_int8(bin[i], m[i], v[i], channel, block_len);
    }
  }

  auto mt = at::mean(mean_t, 1);
  auto mt2 = at::mean(var_t, 1);
  auto vt = mt2 - mt * mt;
  auto* mt_ptr = mt.data_ptr();
  auto* vt_ptr = vt.data_ptr();
  mt = mt.reshape({channel});  // batch = 1 in inference
  vt = vt.reshape({channel});

  at::Tensor scale, shift;
  auto scale_temp = weight / at::sqrt(vt + 1e-5);
  scale = scale_temp;
  shift = (bias - mt * scale_temp);
  auto* scale_ptr = scale.data_ptr();
  auto* shift_ptr = shift.data_ptr();

  auto dil_output = input;
  auto* out_ptr = dil_output.get_data_handle();

  // calculate norm
  if (data_type == dil::data_type::f32) {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<float (*)[block_len * channel]>(in_ptr);
      auto* w = reinterpret_cast<float (*)>(w_ptr);
      auto* b = reinterpret_cast<float (*)>(b_ptr);
      auto* m = reinterpret_cast<float (*)[channel]>(mt_ptr);
      auto* v = reinterpret_cast<float (*)[channel]>(vt_ptr);
      auto* bout = reinterpret_cast<float (*)[block_len * channel]>(out_ptr);
      i_instancenorm_tpp<16>::calc_norm(bin[i], bout[i], w, b, m[i/block_num], v[i/block_num], channel, block_len);
    }
  } else if (data_type == dil::data_type::u8 || data_type == dil::data_type::s8) {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<int8_t (*)[block_len * channel]>(in_ptr);
      auto* sc = reinterpret_cast<float (*)>(scale_ptr);
      auto* sh = reinterpret_cast<float (*)>(shift_ptr);
      auto* bout = reinterpret_cast<int8_t (*)[block_len * channel]>(out_ptr);
      i_instancenorm_tpp<16>::calc_norm_int8_new(bin[i], bout[i], sc, sh, channel, block_len, i_scale, o_scale);
    }  
  }
  return dil_output;
}

std::vector<at::Tensor> dil_inst_mean_var_clast (const dil::tensor& input) 
{
  auto in_sz = input.get_dims();
  auto batch = in_sz[0];
  auto channel = in_sz[1];
  auto block_len = in_sz[2];
  auto reduce_l = in_sz[2] * in_sz[3] * in_sz[4];
  auto block_num = reduce_l / block_len;
  auto mean_t = at::empty({batch, block_num, channel}, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto var_t = at::empty({batch, block_num, channel}, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto* in_ptr = input.get_data_handle();
  auto* m_ptr = mean_t.data_ptr();
  auto* v_ptr = var_t.data_ptr();
  auto data_type = input.get_data_type();

  // calculate mean and variance
  if (data_type == dil::data_type::f32) {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<float (*)[block_len * channel]>(in_ptr);
      auto* m = reinterpret_cast<float (*)[channel]>(m_ptr);
      auto* v = reinterpret_cast<float (*)[channel]>(v_ptr);
      i_instancenorm_tpp<16>::calc_mean_var(bin[i], m[i], v[i], channel, block_len);
    }
  } else if (data_type == dil::data_type::u8 || data_type == dil::data_type::s8)
  {
    #   pragma omp parallel for
    for (auto i = 0; i < batch * block_num; ++i)
    {
      auto* bin = reinterpret_cast<int8_t (*)[block_len * channel]>(in_ptr);
      auto* m = reinterpret_cast<float (*)[channel]>(m_ptr);
      auto* v = reinterpret_cast<float (*)[channel]>(v_ptr);
      i_instancenorm_tpp<16>::calc_mean_var_int8(bin[i], m[i], v[i], channel, block_len);
    }
  }

  auto mt = at::mean(mean_t, 1);
  auto mt2 = at::mean(var_t, 1);
  auto vt = mt2 - mt * mt; 
  return {mt, vt};
}

template <>
void i_instancenorm_tpp<16>::ref(float *in, float &m, float &v, int64_t rl) 
{
  int64_t d;
  auto vsum = _mm512_setzero_ps();
  auto vsum2 = _mm512_setzero_ps();

  auto* pin = in;

  // Pass 1, statistics
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_ps(&pin[d]);
    auto s = f;
    auto ss = s * s;
    vsum += s;
    vsum2 += ss;
  }
  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1<<rem) -1;
    auto zeros = _mm512_setzero_ps();
    auto f = _mm512_mask_loadu_ps(zeros, k, &pin[d]);
    auto s = f;
    auto ss = s * s;
    vsum += s;
    vsum2 += ss;
  }

  auto vmean = _mm512_mean_reduce_ps(vsum, rl);
  auto vmean2 = _mm512_mean_reduce_ps(vsum2, rl);
  auto vvar2 =  vmean2 - vmean * vmean;

  m = vmean[0];
  v = vvar2[0];
}

template <>
void i_instancenorm_tpp<16>::ref(int8_t *in, float &m, float &v, int64_t rl) 
{
  int64_t d;
  auto vsum = _mm512_setzero_ps();
  auto vsum2 = _mm512_setzero_ps();

  auto pin = in;

  // Pass 1
  for (d = 0; d < rl / 16 * 16; d += 16) {
    auto f = _mm512_loadu_i8_to_fp32(&pin[d]);
    auto s = f;
    auto ss = s * s;

    vsum += s;
    vsum2 += ss;
  }
  // Tail
  if (d < rl) {
    auto rem = rl - d;
    __mmask16 k = (1<<rem) -1;
    auto zeros = _mm_setzero_si128();
    auto f = _mm512_mask_loadu_i8_to_fp32(zeros, k, &pin[d]);
    auto s = f;
    auto ss = s * s;

    vsum += s;
    vsum2 += ss;
  }

  auto vmean = _mm512_mean_reduce_ps(vsum, rl);
  auto vmean2 = _mm512_mean_reduce_ps(vsum2, rl);
  auto vvar2 =  vmean2 - vmean * vmean;
  m = vmean[0];
  v = vvar2[0];
}


std::vector<at::Tensor> inst_mean_var (const at::Tensor& input) 
{
  auto in_sz = input.sizes();
  auto batch = in_sz[0] * in_sz[1];
  auto reduce_l = in_sz[2] * in_sz[3] * in_sz[4];
  auto mean_t = at::empty(batch, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto var_t = at::empty(batch, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto* in = input.data_ptr();
  auto* mean = mean_t.data_ptr();
  auto* var = var_t.data_ptr();
  auto data_type = input.scalar_type();

  if (data_type == c10::ScalarType::Char) {
#   pragma omp parallel for
    for (auto i = 0; i < batch; ++i) {
      auto* pin = reinterpret_cast<int8_t (*)[reduce_l]>(in);
      auto* pm = reinterpret_cast<float *>(mean);
      auto* pv = reinterpret_cast<float *>(var);

      i_instancenorm_tpp<16>::ref(pin[i], pm[i], pv[i], reduce_l);
    }
  } else if (data_type == c10::ScalarType::Float) {
#   pragma omp parallel for
    for (auto i = 0; i < batch; ++i) {
      auto* pin = reinterpret_cast<float (*)[reduce_l]>(in);
      auto* pm = reinterpret_cast<float *>(mean);
      auto* pv = reinterpret_cast<float *>(var);

      i_instancenorm_tpp<16>::ref(pin[i], pm[i], pv[i], reduce_l);
    }
  } // throw here

  std::vector<at::Tensor> output;
  output.push_back(mean_t);
  output.push_back(var_t);
  return output;
}

std::vector<at::Tensor> dil_inst_mean_var (const dil::tensor& input) 
{
  auto in_sz = input.get_dims();
  auto batch = in_sz[0] * in_sz[1];
  auto reduce_l = in_sz[2] * in_sz[3] * in_sz[4];
  auto mean_t = at::empty(batch, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));
  auto var_t = at::empty(batch, at::TensorOptions().dtype<float>().memory_format(c10::MemoryFormat::Contiguous));

  auto* in = input.get_data_handle();
  auto* mean = mean_t.data_ptr();
  auto* var = var_t.data_ptr();
  auto data_type = input.get_data_type();

  if (data_type == dil::data_type::u8 || data_type == dil::data_type::s8) {
#   pragma omp parallel for
    for (auto i = 0; i < batch; ++i) {
      auto* pin = reinterpret_cast<int8_t (*)[reduce_l]>(in);
      auto* pm = reinterpret_cast<float *>(mean);
      auto* pv = reinterpret_cast<float *>(var);

      i_instancenorm_tpp<16>::ref(pin[i], pm[i], pv[i], reduce_l);
    }
  } else if (data_type == dil::data_type::f32) {
#   pragma omp parallel for
    for (auto i = 0; i < batch; ++i) {
      auto* pin = reinterpret_cast<float (*)[reduce_l]>(in);
      auto* pm = reinterpret_cast<float *>(mean);
      auto* pv = reinterpret_cast<float *>(var);

      i_instancenorm_tpp<16>::ref(pin[i], pm[i], pv[i], reduce_l);
    }
  } // throw here

  std::vector<at::Tensor> output;
  output.push_back(mean_t);
  output.push_back(var_t);
  return output;
}

template <>
void i_instancenorm_tpp<16>::ref_concat(int8_t *in1, int8_t *in2, int8_t *out, int64_t c, int64_t bl) 
{
  auto* pin1 = in1;
  auto* pin2 = in2;
  auto* pout = out;
  int len = 64;
  auto row = bl;
  auto col = c / len;

  for (auto i = 0; i < row; ++i)
  {
    for (auto j = 0; j < col; ++j)
    {
      auto f1 = _mm512_loadu_si512((__m512i *)&pin1[i * c + j * len]);
      auto f2 = _mm512_loadu_si512((__m512i *)&pin2[i * c + j * len]);
      _mm512_storeu_si512((__m512i *)&pout[i * 2 * c + j * len], f1);
      _mm512_storeu_si512((__m512i *)&pout[i * 2 * c + j * len + c], f2);
    }
  }
}

template <>
void i_instancenorm_tpp<16>::ref_concat32(int8_t *in1, int8_t *in2, int8_t *out, int64_t c, int64_t bl) 
{
  auto* pin1 = in1;
  auto* pin2 = in2;
  auto* pout = out;
  int len = 16;
  auto row = bl;
  auto col = c / len;

  for (auto i = 0; i < row; ++i)
  {
    for (auto j = 0; j < col; ++j)
    {
      auto f1 = _mm_loadu_si128((__m128i *)&pin1[i * c + j * len]);
      auto f2 = _mm_loadu_si128((__m128i *)&pin2[i * c + j * len]);
      _mm_storeu_si128((__m128i *)&pout[i * 2 * c + j * len], f1);
      _mm_storeu_si128((__m128i *)&pout[i * 2 * c + j * len + c], f2);
    }
  }
}

void my_concat(const dil::tensor& input1, const dil::tensor& input2, dil::tensor& output, int dim)
{
#if defined(IPEX_PROFILE_OP)
RECORD_FUNCTION("my_concat", std::vector<c10::IValue>({}));
#endif

  auto in_sz = input1.get_dims();
  auto channel = in_sz[1];
  auto reduce_l = in_sz[0] * in_sz[2] * in_sz[3] * in_sz[4];
  auto block_len = in_sz[2];
  auto batch = reduce_l / block_len;

  auto* in_p1 = input1.get_data_handle();
  auto* in_p2 = input2.get_data_handle();
  auto* d_ptr = output.get_data_handle();

  #   pragma omp parallel for
  for (auto i = 0; i < batch; ++i)
  {
      auto* bin1 = reinterpret_cast<int8_t (*)[block_len * channel]>(in_p1);
      auto* bin2 = reinterpret_cast<int8_t (*)[block_len * channel]>(in_p2);
      auto* bout = reinterpret_cast<int8_t (*)[block_len * channel * 2]>(d_ptr);
      i_instancenorm_tpp<16>::ref_concat32(bin1[i], bin2[i], bout[i], channel, block_len);
  }
}

}
