#pragma once

// Host reference implementations of the FlagGems kernels the core L2 op
// adapter calls, written against the kernel contracts (argument order,
// Triton program id, BLOCK). Test-only: used by the hostsim runtime and by
// the AMD runtime check to evaluate launches on the host.
//
// GGUF block layouts follow ggml-common.h; this file does not include GGML.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace flagos_ref {

struct program {
    uint32_t pid_x;
    uint32_t pid_y;
    uint32_t pid_z;
    uint32_t block;           // Triton BLOCK of the kernel
    uint32_t tile_n;          // columns per program for tiled kernels, 0 if unused
    uint32_t tile_m;          // rows per program for tiled kernels, 0 if unused
    void * const * args;      // pointers to argument values, as in hipModuleLaunchKernel
};

template <typename T> inline T arg(const program & p, int i) {
    T v;
    std::memcpy(&v, p.args[i], sizeof(T));
    return v;
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            mant &= 0x3ff;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000;
    const int32_t exp = int32_t((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00 | (mant ? 0x200 : 0));
    if (exp >= 31) return uint16_t(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return uint16_t(sign);
        mant |= 0x800000;
        const uint32_t shift = uint32_t(14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1);
        const uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = sign | (uint32_t(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
    return uint16_t(half);
}

inline float f16_at(const uint8_t * p) {
    uint16_t h;
    std::memcpy(&h, p, sizeof(h));
    return f16_to_f32(h);
}

// --- GGUF dequantization of one block into `out` ---------------------------

inline void dequant_q4_0(const uint8_t * b, float * out) {   // 18 bytes / 32
    const float d = f16_at(b);
    for (int j = 0; j < 16; ++j) {
        out[j] = (float((b[2 + j] & 0x0f)) - 8.0f) * d;
        out[j + 16] = (float((b[2 + j] >> 4)) - 8.0f) * d;
    }
}

inline void dequant_q4_1(const uint8_t * b, float * out) {   // 20 bytes / 32
    const float d = f16_at(b);
    const float m = f16_at(b + 2);
    for (int j = 0; j < 16; ++j) {
        out[j] = float(b[4 + j] & 0x0f) * d + m;
        out[j + 16] = float(b[4 + j] >> 4) * d + m;
    }
}

inline void dequant_q8_0(const uint8_t * b, float * out) {   // 34 bytes / 32
    const float d = f16_at(b);
    for (int j = 0; j < 32; ++j) {
        out[j] = float(int8_t(b[2 + j])) * d;
    }
}

inline void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

inline void dequant_q4_k(const uint8_t * b, float * out) {   // 144 bytes / 256
    const float d = f16_at(b);
    const float dmin = f16_at(b + 2);
    const uint8_t * scales = b + 4;
    const uint8_t * q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) *out++ = d1 * (q[l] & 0xf) - m1;
        for (int l = 0; l < 32; ++l) *out++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

inline void dequant_q5_k(const uint8_t * b, float * out) {   // 176 bytes / 256
    const float d = f16_at(b);
    const float dmin = f16_at(b + 2);
    const uint8_t * scales = b + 4;
    const uint8_t * qh = b + 16;
    const uint8_t * ql = b + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc, m1 = dmin * m;
        scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) *out++ = d1 * ((ql[l] & 0xf) + (qh[l] & u1 ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *out++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

inline void dequant_q6_k(const uint8_t * b, float * out) {   // 210 bytes / 256
    const uint8_t * ql = b;
    const uint8_t * qh = b + 128;
    const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192);
    const float d = f16_at(b + 208);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = int((ql[l + 0] & 0xf) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = int((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l + 0] = d * sc[is + 0] * q1;
            out[l + 32] = d * sc[is + 2] * q2;
            out[l + 64] = d * sc[is + 4] * q3;
            out[l + 96] = d * sc[is + 6] * q4;
        }
        out += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

struct quant_format {
    int block_elems;
    int block_bytes;
    void (*dequant)(const uint8_t *, float *);
};

inline bool quant_format_of(const std::string & name, quant_format & f) {
    if (name.find("q4_0") != std::string::npos) { f = { 32, 18, dequant_q4_0 }; return true; }
    if (name.find("q4_1") != std::string::npos) { f = { 32, 20, dequant_q4_1 }; return true; }
    if (name.find("q8_0") != std::string::npos) { f = { 32, 34, dequant_q8_0 }; return true; }
    if (name.find("q4_k") != std::string::npos) { f = { 256, 144, dequant_q4_k }; return true; }
    if (name.find("q5_k") != std::string::npos) { f = { 256, 176, dequant_q5_k }; return true; }
    if (name.find("q6_k") != std::string::npos) { f = { 256, 210, dequant_q6_k }; return true; }
    return false;
}

// y[row, col] = dot(dequant(W[row, :]), x[col, :]); W is rows x k quantized.
inline void quant_dot_rows(const quant_format & f, const uint8_t * w, const float * x, float * y,
                           int32_t k, int32_t rows, int64_t row, int64_t col) {
    const int64_t blocks = k / f.block_elems;
    const uint8_t * wr = w + row * blocks * f.block_bytes;
    float tmp[256];
    double sum = 0.0;
    for (int64_t b = 0; b < blocks; ++b) {
        f.dequant(wr + b * f.block_bytes, tmp);
        const float * xb = x + col * int64_t(k) + b * f.block_elems;
        for (int i = 0; i < f.block_elems; ++i) sum += double(tmp[i]) * xb[i];
    }
    y[col * int64_t(rows) + row] = float(sum);
}

// --- program execution -----------------------------------------------------

inline void range(const program & p, int32_t n, int64_t & b, int64_t & e) {
    b = int64_t(p.pid_x) * p.block;
    e = std::min<int64_t>(b + p.block, n);
}

// Run one program of kernel `name`. Returns false for an unknown kernel.
inline bool run(const std::string & name, const program & p) {
    auto F = [&](int i) { return arg<float *>(p, i); };
    auto I = [&](int i) { return arg<int32_t>(p, i); };
    auto S = [&](int i) { return arg<float>(p, i); };
    int64_t b, e;
    if (name == "flagos_add_f32") { range(p, I(3), b, e); for (int64_t i = b; i < e; ++i) F(2)[i] = F(0)[i] + F(1)[i]; return true; }
    if (name == "flagos_add_repeat_f32") { range(p, I(3), b, e); for (int64_t i = b; i < e; ++i) F(2)[i] = F(0)[i] + F(1)[i % I(4)]; return true; }
    if (name == "flagos_mul_f32") { range(p, I(3), b, e); for (int64_t i = b; i < e; ++i) F(2)[i] = F(0)[i] * F(1)[i % I(4)]; return true; }
    if (name == "flagos_scale_f32") { range(p, I(4), b, e); for (int64_t i = b; i < e; ++i) F(1)[i] = F(0)[i] * S(2) + S(3); return true; }
    if (name == "flagos_silu_f32") { range(p, I(2), b, e); for (int64_t i = b; i < e; ++i) F(1)[i] = F(0)[i] / (1.0f + std::exp(-F(0)[i])); return true; }
    if (name == "flagos_sigmoid_f32") { range(p, I(2), b, e); for (int64_t i = b; i < e; ++i) F(1)[i] = 1.0f / (1.0f + std::exp(-F(0)[i])); return true; }
    if (name == "flagos_tanh_f32") { range(p, I(2), b, e); for (int64_t i = b; i < e; ++i) F(1)[i] = std::tanh(F(0)[i]); return true; }
    if (name == "flagos_rms_norm_f32") {
        const int32_t n = I(2);
        const float * x = F(1) + int64_t(p.pid_x) * n;
        float * y = F(0) + int64_t(p.pid_x) * n;
        double sum = 0.0;
        for (int32_t i = 0; i < n; ++i) sum += double(x[i]) * x[i];
        const float s = 1.0f / std::sqrt(float(sum / n) + S(3));
        for (int32_t i = 0; i < n; ++i) y[i] = x[i] * s;
        return true;
    }
    if (name == "flagos_softplus_f32") { range(p, I(2), b, e); for (int64_t i = b; i < e; ++i) { const float v = F(0)[i]; F(1)[i] = v > 20.0f ? v : std::log(1.0f + std::exp(v)); } return true; }
    if (name == "flagos_swiglu_split_f32") { range(p, I(3), b, e); for (int64_t i = b; i < e; ++i) { const float g = F(0)[i]; F(2)[i] = g / (1.0f + std::exp(-g)) * F(1)[i]; } return true; }
    if (name == "flagos_concat_f32") {   // (a, b, out, dim, ye[4], ae[4], as[4], bs[4], n)
        range(p, I(20), b, e);
        const int dim = I(3);
        for (int64_t o = b; o < e; ++o) {
            int64_t y[4], r = o;
            for (int d = 0; d < 4; ++d) { y[d] = r % I(4 + d); r /= I(4 + d); }
            const int64_t aext = I(8 + dim);
            if (y[dim] < aext) {
                F(2)[o] = F(0)[y[0] * I(12) + y[1] * I(13) + y[2] * I(14) + y[3] * I(15)];
            } else {
                y[dim] -= aext;
                F(2)[o] = F(1)[y[0] * I(16) + y[1] * I(17) + y[2] * I(18) + y[3] * I(19)];
            }
        }
        return true;
    }
    if (name == "flagos_l2_norm_strided_f32") {   // (x, out, n_cols, s1, s2, s3, eps, ne1, ne2); grid (ne1, ne2, ne3)
        const int64_t row = p.pid_x, ch = p.pid_y, smp = p.pid_z;
        const int32_t n = I(2);
        const float * x = F(0) + row * I(3) + ch * I(4) + smp * I(5);
        float sum = 0.0f;
        for (int32_t c = 0; c < n; ++c) sum += x[c] * x[c];
        const float scale = 1.0f / std::max(std::sqrt(sum), S(6));
        float * out = F(1) + ((smp * I(8) + ch) * I(7) + row) * n;
        for (int32_t c = 0; c < n; ++c) out[c] = x[c] * scale;
        return true;
    }
    if (name == "flagos_rms_norm_ggml_strided_f32") {   // (out, x, n_cols, ne1, ne2, s1, s2, s3, eps)
        const int32_t n = I(2), ne1 = I(3), ne2 = I(4);
        const int64_t row = p.pid_x;
        const int64_t i1 = row % ne1, i2 = (row / ne1) % ne2, i3 = row / (int64_t(ne1) * ne2);
        const float * x = F(1) + i1 * I(5) + i2 * I(6) + i3 * I(7);
        double sum = 0.0;
        for (int32_t c = 0; c < n; ++c) { const float sq = x[c] * x[c]; sum += sq; }
        const float mean = float(sum / n);
        const float scale = 1.0f / std::sqrt(mean + S(8));
        for (int32_t c = 0; c < n; ++c) F(0)[row * n + c] = x[c] * scale;
        return true;
    }
    if (name == "flagos_ssm_conv_f32") {   // (s, c, out, d_conv, d_inner, n_tokens, n_seqs, ss1, ss2, sc1, so0, so1)
        const int64_t t = p.pid_x, seq = p.pid_y;
        const int32_t d_conv = I(3), d_inner = I(4);
        const int64_t c0 = int64_t(p.pid_z) * p.block, c1 = std::min<int64_t>(c0 + p.block, d_inner);
        for (int64_t ch = c0; ch < c1; ++ch) {
            float acc = 0.0f;
            for (int32_t tap = 0; tap < d_conv; ++tap) acc += F(0)[seq * I(8) + ch * I(7) + t + tap] * F(1)[ch * I(9) + tap];
            F(2)[seq * I(11) + t * I(10) + ch] = acc;
        }
        return true;
    }
    if (name == "flagos_gated_delta_net_scalar_f32") {
        // (q, k, v, g, beta, state, out, Sd, H, T, B, sq1..3, sv1..3, sb1..3, q_heads, q_seq_ratio, K, scale)
        const int32_t Sd = I(7), H = I(8), T = I(9), B = I(10);
        const int32_t sq1 = I(11), sq2 = I(12), sq3 = I(13), sv1 = I(14), sv2 = I(15), sv3 = I(16);
        const int32_t sb1 = I(17), sb2 = I(18), sb3 = I(19), q_heads = I(20), ratio = I(21), K = I(22);
        const float scale = arg<float>(p, 23);
        const int64_t head = p.pid_x, seq = p.pid_y;
        const int64_t cols = p.tile_n;
        const int64_t j0 = int64_t(p.pid_z) * cols, j1 = std::min<int64_t>(j0 + cols, Sd);
        const int64_t qh = head % q_heads, qs = seq / ratio;
        const int64_t sb = (seq * H + head) * int64_t(Sd) * Sd;
        const int64_t attn_elems = int64_t(Sd) * H * T * B, snap_stride = int64_t(Sd) * Sd * H * B;
        const float * q = F(0); const float * k = F(1); const float * v = F(2);
        const float * g = F(3); const float * beta = F(4); const float * state = F(5); float * out = F(6);
        std::vector<float> M(size_t(j1 - j0) * Sd);
        for (int64_t j = j0; j < j1; ++j) for (int32_t i = 0; i < Sd; ++i) M[(j - j0) * Sd + i] = state[sb + j * Sd + i];
        for (int32_t t = 0; t < T; ++t) {
            const int64_t qb = qs * sq3 + int64_t(t) * sq2 + qh * sq1;
            const int64_t gb = seq * sb3 + int64_t(t) * sb2 + head * sb1;
            const float decay = std::exp(g[gb]), bt = beta[gb];
            for (int64_t j = j0; j < j1; ++j) {
                float * m = &M[(j - j0) * Sd];
                float pk = 0.0f;
                for (int32_t i = 0; i < Sd; ++i) { m[i] *= decay; pk += m[i] * k[qb + i]; }
                const float d = (v[seq * sv3 + int64_t(t) * sv2 + head * sv1 + j] - pk) * bt;
                float a = 0.0f;
                for (int32_t i = 0; i < Sd; ++i) { m[i] += d * k[qb + i]; a += m[i] * q[qb + i]; }
                out[((seq * T + t) * H + head) * Sd + j] = a * scale;
                const int32_t slot = T - 1 - t;
                if (slot < K) for (int32_t i = 0; i < Sd; ++i) out[attn_elems + slot * snap_stride + sb + j * Sd + i] = m[i];
            }
        }
        return true;
    }
    if (name == "flagos_rope_ggml_recurrence_f32" || name == "flagos_rope_ggml_neox_recurrence_f32") {
        // (x, pos, factors, out, ne0, ne1, ne2, n_dims, mode, has_factors, sx1, sx2, sx3, s0..s3,
        //  theta_scale, freq_scale, attn_factor); one program per row, BLOCK pairs
        const bool neox_kernel = name == "flagos_rope_ggml_neox_recurrence_f32";
        const int32_t ne0 = I(4), ne1 = I(5), ne2 = I(6), n_dims = I(7), mode = I(8), has_factors = I(9);
        const int32_t st = I(13), sh = I(14), sw = I(15), se = I(16);
        const float theta_scale = arg<float>(p, 17), freq_scale = arg<float>(p, 18), attn_factor = arg<float>(p, 19);
        const float * x = F(0);
        const auto * positions = arg<const int32_t *>(p, 1);
        const float * factors = F(2);
        float * out = F(3);
        const int64_t row = p.pid_x, head = row % ne1, token = (row / ne1) % ne2, batch = row / (int64_t(ne1) * ne2);
        const int64_t src = head * I(10) + token * I(11) + batch * I(12);
        for (int32_t pr = 0; pr < int32_t(p.block) && pr < ne0 / 2; ++pr) {
            const bool rotated = pr < n_dims / 2;
            int32_t first = 2 * pr, second = 2 * pr + 1, axis = 0;
            if (rotated && (neox_kernel || mode == 40)) { first = pr; second = pr + n_dims / 2; }
            if (!neox_kernel && mode == 40) {
                const int32_t sect = pr % (st + sh + sw + se);
                axis = (sect % 3 == 1 && sect < 3 * sh) ? 1 : (sect % 3 == 2 && sect < 3 * sw) ? 2
                     : (sect % 3 == 0 && sect < 3 * st) ? 0 : 3;
            }
            float theta = float(positions[token + int64_t(axis) * ne2]);
            for (int32_t i = 1; i < n_dims / 2; ++i) if (pr >= i) theta *= theta_scale;
            if (has_factors) theta = theta / (rotated ? factors[pr] : 1.0f);
            theta *= freq_scale;
            const float cs = std::cos(theta) * attn_factor, sn = std::sin(theta) * attn_factor;
            const float a = x[src + first], bv = x[src + second];
            out[row * ne0 + first] = rotated ? a * cs - bv * sn : a;
            out[row * ne0 + second] = rotated ? a * sn + bv * cs : bv;
        }
        return true;
    }
    if (name.rfind("flagos_flash_attn_", 0) == 0) {
        // decode:  (q, k, v, mask, out, klen, gqa, sqt, sqh, skt, skh, svt, svh, sot, soh, scale)
        // prefill: (q, k, v, mask, out, qlen, klen, gqa, sqt, sqh, skt, skh, svt, svh, smt, smq, sot, soh, scale)
        const bool prefill = name.find("_prefill_") != std::string::npos;
        const int D = name.find("_h256") != std::string::npos ? 256 : 128;
        int a = 5;
        const int32_t qlen = prefill ? I(a++) : 1;
        const int32_t klen = I(a++), gqa = I(a++), sqt = I(a++), sqh = I(a++), skt = I(a++), skh = I(a++);
        const int32_t svt = I(a++), svh = I(a++);
        const int32_t smt = prefill ? I(a++) : 1, smq = prefill ? I(a++) : 0;
        const int32_t sot = I(a++), soh = I(a++);
        const float scale = arg<float>(p, a);
        const float * q = F(0);
        const auto * k = arg<const uint16_t *>(p, 1);
        const auto * v = arg<const uint16_t *>(p, 2);
        const auto * mask = arg<const uint16_t *>(p, 3);
        float * out = F(4);
        const int64_t qh = p.pid_x, kvh = qh / gqa;
        const int64_t r0 = prefill ? int64_t(p.pid_y) * p.block : 0, r1 = prefill ? std::min<int64_t>(r0 + p.block, qlen) : 1;
        auto f16r = [](float x) { return f16_to_f32(f32_to_f16(x)); };
        std::vector<float> Q(D), acc(D);
        for (int64_t r = r0; r < r1; ++r) {
            for (int d = 0; d < D; ++d) Q[d] = f16r(q[r * sqt + qh * sqh + d]);
            float m = -INFINITY, l = 0.0f;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (int64_t ks = 0; ks < klen; ks += 16) {
                float S[16], P[16];
                float bm = -INFINITY;
                for (int n = 0; n < 16; ++n) {
                    const int64_t key = ks + n;
                    if (key >= klen) { S[n] = -INFINITY; continue; }
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) dot += Q[d] * f16_to_f32(k[d + kvh * skh + key * skt]);
                    S[n] = dot * scale + f16_to_f32(mask[key * smt + (prefill ? r * smq : 0)]);
                    bm = std::max(bm, S[n]);
                }
                const float nm = std::max(m, bm);
                const float safe = nm == -INFINITY ? 0.0f : nm;
                const float resc = std::exp(m - safe);
                float psum = 0.0f;
                for (int n = 0; n < 16; ++n) { P[n] = S[n] == -INFINITY ? 0.0f : std::exp(S[n] - safe); psum += P[n]; }
                l = l * resc + psum;
                for (int d = 0; d < D; ++d) {
                    float t = 0.0f;
                    for (int n = 0; n < 16; ++n) {
                        if (ks + n < klen) t += f16r(P[n]) * f16_to_f32(v[(ks + n) * svt + kvh * svh + d]);
                    }
                    acc[d] = acc[d] * resc + t;
                }
                m = nm;
            }
            for (int d = 0; d < D; ++d) out[r * sot + qh * soh + d] = acc[d] / (l > 0.0f ? l : 1.0f);
        }
        return true;
    }
    if (name == "flagos_soft_max_unmasked_f32" || name == "flagos_soft_max_masked_f32_f16") {
        // (x, mask|null, out, n_cols, scale); one program per row
        const int32_t n = I(3);
        const float scale = S(4);
        const float * x = F(0) + int64_t(p.pid_x) * n;
        const auto * mask = arg<const uint16_t *>(p, 1);
        float * out = F(2) + int64_t(p.pid_x) * n;
        std::vector<float> v(n);
        float m = -INFINITY;
        for (int32_t c = 0; c < n; ++c) {
            v[c] = x[c] * scale + (mask != nullptr ? f16_to_f32(mask[int64_t(p.pid_x) * n + c]) : 0.0f);
            m = std::max(m, v[c]);
        }
        float sum = 0.0f;
        for (int32_t c = 0; c < n; ++c) { v[c] = std::exp(v[c] - m); sum += v[c]; }
        for (int32_t c = 0; c < n; ++c) out[c] = v[c] / sum;
        return true;
    }
    if (name.rfind("flagos_get_rows_q", 0) == 0) {   // (w_u8, w_f16, idx, out, n_cols); grid (rows, n_cols / QK)
        quant_format f;
        if (!quant_format_of(name, f)) return false;
        const int32_t n = I(4);
        const auto * w = arg<const uint8_t *>(p, 0);
        const auto * idx = arg<const int32_t *>(p, 2);
        float * out = F(3);
        const int64_t row = idx[p.pid_x];
        const int64_t blk = row * (n / f.block_elems) + p.pid_y;
        f.dequant(w + blk * f.block_bytes, out + int64_t(p.pid_x) * n + int64_t(p.pid_y) * f.block_elems);
        return true;
    }
    if (name.rfind("flagos_argsort_f32_i32_b", 0) == 0) {   // (x, out, columns, row_stride, descending)
        const int32_t cols = I(2), stride = I(3), desc = I(4);
        const float * x = F(0) + int64_t(p.pid_x) * stride;
        auto * out = arg<int32_t *>(p, 1) + int64_t(p.pid_x) * cols;
        std::vector<int32_t> idx(cols);
        for (int32_t i = 0; i < cols; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](int32_t i, int32_t j) {
            const float a = x[i], bv = x[j];
            if (std::isnan(a) || std::isnan(bv)) return !std::isnan(a) && std::isnan(bv);
            return desc ? a > bv : a < bv;
        });
        for (int32_t i = 0; i < cols; ++i) out[i] = idx[i];
        return true;
    }
    if (name == "flagos_sum_rows_f32_acc_f64") {   // (x, out, n_cols)
        const int32_t n = I(2);
        const float * x = F(0) + int64_t(p.pid_x) * n;
        double sum = 0.0;
        for (int32_t i = 0; i < n; ++i) sum += x[i];
        F(1)[p.pid_x] = float(sum);
        return true;
    }
    if (name == "flagos_clamp_compare_f32") {   // (x, out, n, min, max)
        range(p, I(2), b, e);
        for (int64_t i = b; i < e; ++i) { float v = F(0)[i]; v = v < S(4) ? v : S(4); v = v > S(3) ? v : S(3); F(1)[i] = v; }
        return true;
    }
    if (name == "flagos_add_repeat_strided_f32" || name == "flagos_mul_repeat_strided_f32" ||
        name == "flagos_div_repeat_strided_f32") {   // (x, y, out, n[4], y[4], sx[4], sy[4], n_elements)
        range(p, I(19), b, e);
        const char kind = name[7];   // 'a', 'm', 'd'
        for (int64_t o = b; o < e; ++o) {
            int64_t r = o, ix = 0, iy = 0;
            for (int d = 0; d < 4; ++d) {
                const int64_t c = r % I(3 + d);
                r /= I(3 + d);
                ix += c * I(11 + d);
                iy += (c % I(7 + d)) * I(15 + d);
            }
            const float av = F(0)[ix], bv2 = F(1)[iy];
            F(2)[o] = kind == 'a' ? av + bv2 : kind == 'm' ? av * bv2 : av / bv2;
        }
        return true;
    }
    if (name == "flagos_get_rows_batched_f32") {
        // (table, idx, out, n_cols, table_rows, gather_rows, batch2, batch3, st0..3, si0..2, n)
        range(p, I(15), b, e);
        const int64_t cols = I(3), trows = I(4), grows = I(5), b2n = I(6);
        const auto * idx = arg<const int32_t *>(p, 1);
        for (int64_t o = b; o < e; ++o) {
            const int64_t col = o % cols, tok = (o / cols) % grows, b2 = (o / (cols * grows)) % b2n, b3 = o / (cols * grows * b2n);
            const int64_t row = idx[tok * I(12) + b2 * I(13) + b3 * I(14)];
            F(2)[o] = (row >= 0 && row < trows) ? F(0)[col * I(8) + row * I(9) + b2 * I(10) + b3 * I(11)] : 0.0f;
        }
        return true;
    }
    if (name == "flagos_mul_mat_f32_f32_strided_batched_ieee") {
        // (w, x, out, k, rows, columns, heads, head_ratio, sample_ratio, sw1..3, sx1..3); tile 32 x 16
        const int32_t k = I(3), rows = I(4), cols = I(5), heads = I(6);
        const int64_t head = p.pid_z % heads, sample = p.pid_z / heads;
        const int64_t a_base = (head / I(7)) * I(10) + (sample / I(8)) * I(11);
        const int64_t b_base = head * I(13) + sample * I(14);
        for (int64_t m = int64_t(p.pid_x) * 32; m < std::min<int64_t>(int64_t(p.pid_x) * 32 + 32, rows); ++m) {
            for (int64_t n = int64_t(p.pid_y) * 16; n < std::min<int64_t>(int64_t(p.pid_y) * 16 + 16, cols); ++n) {
                float acc = 0.0f;
                for (int32_t kk = 0; kk < k; ++kk) acc += F(0)[a_base + m * I(9) + kk] * F(1)[b_base + n * I(12) + kk];
                F(2)[(sample * heads + head) * int64_t(rows) * cols + n * rows + m] = acc;
            }
        }
        return true;
    }
    if (name.rfind("flagos_mul_mat_id_", 0) == 0) {
        // quantized (17): (w_u8, w_f16, x, ids, out, k, rows, picks, tokens, input_picks, experts, sw1_b, sw2_b, sx1, sx2, si0, si1)
        // f32 (16):       (w, x, ids, out, k, rows, picks, tokens, input_picks, experts, sw1, sw2, sx1, sx2, si0, si1)
        const bool f32w = name == "flagos_mul_mat_id_f32_f32";
        const int a0 = f32w ? 4 : 5;
        const int32_t k = I(a0), rows = I(a0 + 1), picks = I(a0 + 2), input_picks = I(a0 + 4), experts = I(a0 + 5);
        const int64_t sw1 = I(a0 + 6), sw2 = I(a0 + 7), sx1 = I(a0 + 8), sx2 = I(a0 + 9), si0 = I(a0 + 10), si1 = I(a0 + 11);
        const float * x = F(f32w ? 1 : 2);
        const auto * ids = arg<const int32_t *>(p, f32w ? 2 : 3);
        float * out = F(f32w ? 3 : 4);
        const int64_t pick = p.pid_y % picks, token = p.pid_y / picks;
        const int64_t expert = ids[pick * si0 + token * si1];
        const bool valid = expert >= 0 && expert < experts;
        const int64_t xb = token * sx2 + (pick % input_picks) * sx1;
        const int64_t r0 = f32w ? int64_t(p.pid_x) * std::max<uint32_t>(p.tile_m, 1) : p.pid_x;
        const int64_t r1 = f32w ? std::min<int64_t>(r0 + std::max<uint32_t>(p.tile_m, 1), rows) : r0 + 1;
        for (int64_t row = r0; row < r1; ++row) {
            double acc = 0.0;
            if (valid) {
                if (f32w) {
                    const float * wr = F(0) + expert * sw2 + row * sw1;
                    for (int32_t kk = 0; kk < k; ++kk) acc += double(wr[kk]) * x[xb + kk];
                } else {
                    quant_format f;
                    if (!quant_format_of(name, f)) return false;
                    const uint8_t * wr = arg<const uint8_t *>(p, 0) + expert * sw2 + row * sw1;
                    float tmp[256];
                    for (int64_t blk = 0; blk < k / f.block_elems; ++blk) {
                        f.dequant(wr + blk * f.block_bytes, tmp);
                        for (int i = 0; i < f.block_elems; ++i) acc += double(tmp[i]) * x[xb + blk * f.block_elems + i];
                    }
                }
            }
            out[(token * picks + pick) * int64_t(rows) + row] = float(acc);
        }
        return true;
    }
    if (name == "flagos_rms_norm_mul_f32") {   // (norm_out, mul_out, x, weight, n_cols, eps); one program per row
        const int32_t n = I(4);
        const float * x = F(2) + int64_t(p.pid_x) * n;
        const float * w = F(3);
        float * norm = F(0) + int64_t(p.pid_x) * n;
        float * out = F(1) + int64_t(p.pid_x) * n;
        double sum = 0.0;
        for (int32_t i = 0; i < n; ++i) sum += double(x[i]) * x[i];
        const float s = 1.0f / std::sqrt(float(sum / n) + S(5));
        for (int32_t i = 0; i < n; ++i) { norm[i] = x[i] * s; out[i] = norm[i] * w[i]; }
        return true;
    }
    if (name == "flagos_rms_norm_mul_inplace_f32") {   // (out, x, weight, n_cols, eps)
        const int32_t n = I(3);
        const float * x = F(1) + int64_t(p.pid_x) * n;
        const float * w = F(2);
        float * out = F(0) + int64_t(p.pid_x) * n;
        double sum = 0.0;
        for (int32_t i = 0; i < n; ++i) sum += double(x[i]) * x[i];
        const float s = 1.0f / std::sqrt(float(sum / n) + S(4));
        for (int32_t i = 0; i < n; ++i) out[i] = x[i] * s * w[i];
        return true;
    }
    if (name == "flagos_add_rms_norm_mul_residual_f32") {   // (residual, out, x, bias, weight, n_cols, eps)
        const int32_t n = I(5);
        const int64_t base = int64_t(p.pid_x) * n;
        float row[8192];
        if (n > 8192) return false;
        double sum = 0.0;
        for (int32_t i = 0; i < n; ++i) { row[i] = F(2)[base + i] + F(3)[base + i]; sum += double(row[i]) * row[i]; }
        const float s = 1.0f / std::sqrt(float(sum / n) + S(6));
        for (int32_t i = 0; i < n; ++i) { F(0)[base + i] = row[i]; F(1)[base + i] = row[i] * s * F(4)[i]; }
        return true;
    }
    if (name == "flagos_copy_strided_f32") {   // (src, dst, src_ne[4], dst_ne[4], src_stride[4], dst_stride[4], n)
        range(p, I(18), b, e);
        const float * src = F(0);
        float * dst = F(1);
        for (int64_t i = b; i < e; ++i) {
            int64_t r = i, so = 0;
            for (int d = 0; d < 4; ++d) { so += (r % I(2 + d)) * I(10 + d); r /= I(2 + d); }
            r = i;
            int64_t dof = 0;
            for (int d = 0; d < 4; ++d) { dof += (r % I(6 + d)) * I(14 + d); r /= I(6 + d); }
            dst[dof] = src[so];
        }
        return true;
    }
    if (name == "flagos_set_rows_f32_f16") {   // (src, idx_i64, dst_f16, n_cols, n_rows, n_dst_rows)
        const int32_t cols = I(3);
        range(p, cols * I(4), b, e);
        const float * src = F(0);
        const auto * idx = arg<const int64_t *>(p, 1);
        auto * dst = arg<uint16_t *>(p, 2);
        for (int64_t i = b; i < e; ++i) {
            const int64_t row = i / cols, col = i % cols;
            dst[idx[row] * cols + col] = f32_to_f16(src[i]);
        }
        return true;
    }
    if (name == "flagos_get_rows_f32") {       // (table, idx_i32, out, columns); grid (rows, ceil(columns / BLOCK))
        const int32_t cols = I(3);
        const float * table = F(0);
        const auto * idx = arg<const int32_t *>(p, 1);
        float * out = F(2);
        const int64_t c0 = int64_t(p.pid_y) * p.block;
        const int64_t c1 = std::min<int64_t>(c0 + p.block, cols);
        for (int64_t c = c0; c < c1; ++c) out[int64_t(p.pid_x) * cols + c] = table[int64_t(idx[p.pid_x]) * cols + c];
        return true;
    }
    // Dequantize to F16: (w_u8, w_f16, out_f16); one program per quant block.
    if (name.rfind("flagos_dequant_", 0) == 0) {
        quant_format f;
        if (!quant_format_of(name, f)) return false;
        const auto * w = arg<const uint8_t *>(p, 0);
        auto * out = arg<uint16_t *>(p, 2);
        float tmp[256];
        f.dequant(w + int64_t(p.pid_x) * f.block_bytes, tmp);
        for (int i = 0; i < f.block_elems; ++i) out[int64_t(p.pid_x) * f.block_elems + i] = f32_to_f16(tmp[i]);
        return true;
    }
    // F16 GEMM: (w_f16, x, y, k, rows, columns); grid (ceil(rows / tile_m),
    // ceil(columns / tile_n)); activations are rounded to F16, F32 accumulate.
    if (name.rfind("flagos_mul_mat_f16_f32_batched", 0) == 0) {
        const auto * w = arg<const uint16_t *>(p, 0);
        const float * x = F(1);
        float * y = F(2);
        const int32_t k = I(3), rows = I(4), columns = I(5);
        const int64_t r0 = int64_t(p.pid_x) * p.tile_m, r1 = std::min<int64_t>(r0 + p.tile_m, rows);
        const int64_t c0 = int64_t(p.pid_y) * p.tile_n, c1 = std::min<int64_t>(c0 + p.tile_n, columns);
        for (int64_t c = c0; c < c1; ++c) {
            for (int64_t r = r0; r < r1; ++r) {
                double sum = 0.0;
                for (int32_t i = 0; i < k; ++i) {
                    sum += double(f16_to_f32(w[r * k + i])) * f16_to_f32(f32_to_f16(x[c * k + i]));
                }
                y[c * rows + r] = float(sum);
            }
        }
        return true;
    }
    // Quantized GEMV: (w_u8, w_f16, x, y, k, rows); one program per row.
    // Batched: (w_u8, w_f16, x, y, k, rows, columns); grid (rows, ceil(columns / BLOCK)).
    if (name.rfind("flagos_mul_mat_q", 0) == 0) {
        quant_format f;
        if (!quant_format_of(name, f)) return false;
        const auto * w = arg<const uint8_t *>(p, 0);
        const float * x = arg<const float *>(p, 2);
        float * y = arg<float *>(p, 3);
        const int32_t k = I(4);
        const int32_t rows = I(5);
        const bool batched = name.find("_batched") != std::string::npos;
        if (!batched) {
            quant_dot_rows(f, w, x, y, k, rows, p.pid_x, 0);
            return true;
        }
        const int32_t columns = I(6);
        const int64_t c0 = int64_t(p.pid_y) * p.block;
        const int64_t c1 = std::min<int64_t>(c0 + p.block, columns);
        for (int64_t c = c0; c < c1; ++c) quant_dot_rows(f, w, x, y, k, rows, p.pid_x, c);
        return true;
    }
    return false;
}

} // namespace flagos_ref
