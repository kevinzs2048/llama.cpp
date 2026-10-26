#pragma once

// SHA-256 used to verify kernel-package artifacts. Shared by the core
// package reader and provider-side loaders.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace flagos {

struct sha256_context {
    uint32_t state[8];
    uint64_t bit_length;
    uint8_t buffer[64];
    size_t buffer_length;
};

inline constexpr uint32_t SHA256_CONSTANTS[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotate_right(uint32_t value, unsigned int shift) {
    return (value >> shift) | (value << (32 - shift));
}

inline void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t words[64];
    for (int i = 0; i < 16; ++i) {
        words[i] = (uint32_t(block[i * 4]) << 24) |
                   (uint32_t(block[i * 4 + 1]) << 16) |
                   (uint32_t(block[i * 4 + 2]) << 8) |
                    uint32_t(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
        const uint32_t s1 = rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + sum1 + choose + SHA256_CONSTANTS[i] + words[i];
        const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

inline void sha256_initialize(sha256_context & context) {
    context.state[0] = 0x6a09e667;
    context.state[1] = 0xbb67ae85;
    context.state[2] = 0x3c6ef372;
    context.state[3] = 0xa54ff53a;
    context.state[4] = 0x510e527f;
    context.state[5] = 0x9b05688c;
    context.state[6] = 0x1f83d9ab;
    context.state[7] = 0x5be0cd19;
    context.bit_length = 0;
    context.buffer_length = 0;
}

inline void sha256_update(sha256_context & context, const uint8_t * data, size_t length) {
    context.bit_length += uint64_t(length) * 8;
    if (context.buffer_length != 0) {
        const size_t copy_length = std::min(length, sizeof(context.buffer) - context.buffer_length);
        std::memcpy(context.buffer + context.buffer_length, data, copy_length);
        context.buffer_length += copy_length;
        data += copy_length;
        length -= copy_length;
        if (context.buffer_length == sizeof(context.buffer)) {
            sha256_compress(context.state, context.buffer);
            context.buffer_length = 0;
        }
    }
    while (length >= sizeof(context.buffer)) {
        sha256_compress(context.state, data);
        data += sizeof(context.buffer);
        length -= sizeof(context.buffer);
    }
    if (length != 0) {
        std::memcpy(context.buffer, data, length);
        context.buffer_length = length;
    }
}

inline std::string sha256_hex(const std::vector<uint8_t> & bytes) {
    sha256_context context;
    sha256_initialize(context);
    sha256_update(context, bytes.data(), bytes.size());
    const uint64_t bit_length = context.bit_length;
    context.buffer[context.buffer_length++] = 0x80;
    if (context.buffer_length > 56) {
        while (context.buffer_length < sizeof(context.buffer)) {
            context.buffer[context.buffer_length++] = 0;
        }
        sha256_compress(context.state, context.buffer);
        context.buffer_length = 0;
    }
    while (context.buffer_length < 56) {
        context.buffer[context.buffer_length++] = 0;
    }
    for (int i = 7; i >= 0; --i) {
        context.buffer[context.buffer_length++] = uint8_t(bit_length >> (i * 8));
    }
    sha256_compress(context.state, context.buffer);

    static constexpr char HEX[] = "0123456789abcdef";
    std::string result(64, '0');
    for (int i = 0; i < 32; ++i) {
        const uint8_t value = uint8_t(context.state[i / 4] >> (24 - (i % 4) * 8));
        result[i * 2] = HEX[value >> 4];
        result[i * 2 + 1] = HEX[value & 0x0f];
    }
    return result;
}

inline bool is_sha256(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
    });
}

inline std::string sha256_hex(const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    return sha256_hex(std::vector<uint8_t>(bytes, bytes + size));
}

} // namespace flagos
