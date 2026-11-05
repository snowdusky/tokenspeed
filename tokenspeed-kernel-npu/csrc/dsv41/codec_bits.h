// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "kernel_operator.h"
using namespace AscendC;

// 910B lacks the native FP4/FP8 conversions used by FlashMLA on 950.
// Decode sixteen selected, token-interleaved rows at a time using UB lookup tables.
// FP4 values and E4M3 scales multiply exactly into BF16 (for finite values).
struct KvCodec {
    static constexpr int B = 16, D = 512;
    TBuf<> rawBuf, byteBuf, halfBuf, indexBuf, valueBuf, expandedBuf;
    TBuf<> scaleBuf, tablesBuf, pairBuf;
    LocalTensor<uint8_t> raw, bytes;
    LocalTensor<half> halves;
    LocalTensor<int32_t> index;
    LocalTensor<float> values, expanded, scales;
    LocalTensor<uint32_t> pairs, fp4Table;
    LocalTensor<float> e4Table, e8Table;

    __aicore__ inline void Init(TPipe& p, GlobalTensor<uint32_t> table) {
        p.InitBuffer(rawBuf, 2 * B * 544);
        p.InitBuffer(byteBuf, B * D);
        p.InitBuffer(halfBuf, 2 * B * D * 2);
        p.InitBuffer(indexBuf, B * D * 4);
        p.InitBuffer(valueBuf, B * D * 4);
        p.InitBuffer(expandedBuf, B * D * 2);
        p.InitBuffer(scaleBuf, B * 32 * 4);
        p.InitBuffer(tablesBuf, 3 * 256 * 4);
        p.InitBuffer(pairBuf, B * D * 2);
        raw = rawBuf.Get<uint8_t>();
        bytes = byteBuf.Get<uint8_t>();
        halves = halfBuf.Get<half>();
        index = indexBuf.Get<int32_t>();
        values = valueBuf.Get<float>();
        expanded = expandedBuf.Get<float>();
        scales = scaleBuf.Get<float>();
        pairs = pairBuf.Get<uint32_t>();
        fp4Table = tablesBuf.Get<uint32_t>();
        e4Table = tablesBuf.Get<float>()[256];
        e8Table = tablesBuf.Get<float>()[512];
        DataCopy(fp4Table, table, 768);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    // Caller fills raw and orders MTE2 before this call. Invalid rows are zero.
    template <bool FP4>
    __aicore__ inline LocalTensor<bfloat16_t> Decode() {
        constexpr int payload = FP4 ? 256 : 512;
        constexpr int row = FP4 ? 288 : 544;
        // Compact payload bytes; raw scales are padded to a full UB block.
        DataCopy(bytes, raw, DataCopyParams{B, payload / 32, 1, 0});
        PipeBarrier<PIPE_V>();
        Cast(halves, bytes, RoundMode::CAST_NONE, B * payload);
        PipeBarrier<PIPE_V>();
        if constexpr (!FP4) {
            // Interpret E4M3 bits as FP16 with an exponent offset of -8.
            // Scaling by 256 restores every finite code, including subnormals.
            // The two E4M3 NaN codes are handled before applying the sign.
            auto magnitude = indexBuf.Get<uint16_t>();
            auto sign = indexBuf.Get<uint16_t>()[B * D];
            Cast(magnitude.ReinterpretCast<int16_t>(), halves, RoundMode::CAST_RINT, B * D);
            PipeBarrier<PIPE_V>();
            ShiftRight(sign, magnitude, uint16_t(7), B * D);
            ShiftLeft(magnitude, magnitude, uint16_t(9), B * D);
            PipeBarrier<PIPE_V>();
            ShiftLeft(sign, sign, uint16_t(15), B * D);
            ShiftRight(magnitude, magnitude, uint16_t(2), B * D);
            PipeBarrier<PIPE_V>();
            auto decoded = magnitude.ReinterpretCast<half>();
            CompareScalar(bytes, decoded, half(1.875), CMPMODE::NE, B * D);
            PipeBarrier<PIPE_V>();
            union {
                uint16_t bits;
                half value;
            } nan;
            nan.bits = 0x7e00;
            Select(decoded, bytes, decoded, nan.value, SELMODE::VSEL_TENSOR_SCALAR_MODE, B * D);
            PipeBarrier<PIPE_V>();
            Or(magnitude, magnitude, sign, B * D);
            PipeBarrier<PIPE_V>();
            Cast(values, decoded, RoundMode::CAST_NONE, B * D);
            PipeBarrier<PIPE_V>();
            Muls(values, values, 256.f, B * D);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(index, halves, RoundMode::CAST_RINT, B * payload);
            PipeBarrier<PIPE_V>();
            ShiftLeft(index.ReinterpretCast<uint32_t>(), index.ReinterpretCast<uint32_t>(), uint32_t(2), B * payload);
            PipeBarrier<PIPE_V>();
            if constexpr (FP4) {
                Gather(pairs, fp4Table, index.ReinterpretCast<uint32_t>(), 0, B * payload);
                PipeBarrier<PIPE_V>();
            } else {
                Gather(values, e4Table, index.ReinterpretCast<uint32_t>(), 0, B * payload);
            }
            PipeBarrier<PIPE_V>();
        }
        DataCopy(bytes, raw[payload], DataCopyParams{B, 1, (row - 32) / 32, 0});
        PipeBarrier<PIPE_V>();
        Cast(halves, bytes, RoundMode::CAST_NONE, B * 32);
        PipeBarrier<PIPE_V>();
        Cast(index, halves, RoundMode::CAST_RINT, B * 32);
        PipeBarrier<PIPE_V>();
        ShiftLeft(index.ReinterpretCast<uint32_t>(), index.ReinterpretCast<uint32_t>(), uint32_t(2), B * 32);
        PipeBarrier<PIPE_V>();
        Gather(scales, FP4 ? e4Table : e8Table, index.ReinterpretCast<uint32_t>(), 0, B * 32);
        PipeBarrier<PIPE_V>();
        if constexpr (FP4) {
            // E2M1 * finite E4M3 is exactly representable in FP16. Expand one
            // group16 scale to one full vector block, avoiding masked FP32 Mul.
            // Conversion to BF16 still passes through FP32 on 910B.
            Cast(halves, scales, RoundMode::CAST_NONE, B * 32);
            PipeBarrier<PIPE_V>();
            auto expandedHalf = expandedBuf.Get<half>();
            Brcb(expandedHalf, halves, B * 4, {1, 8});
            PipeBarrier<PIPE_V>();
            auto decoded = pairs.ReinterpretCast<half>();
            Mul(decoded, decoded, expandedHalf, B * D);
            PipeBarrier<PIPE_V>();
            Cast(values, decoded, RoundMode::CAST_NONE, B * D);
        } else {
            // Raw FP8 scale rows pad 16 group32 scales to 32 bytes.
            // Remove that gap once, then apply the identical per-group
            // broadcast over two 128-repeat batches instead of 16 row calls.
            auto compactScales = indexBuf.Get<float>();
            DataCopy(compactScales, scales, DataCopyParams{B, 2, 2, 0});
            PipeBarrier<PIPE_V>();
            Brcb(expanded, compactScales, B * 2, {1, 8});
            PipeBarrier<PIPE_V>();
            for (int batch = 0; batch < 2; ++batch)
                Mul(values[batch * (B / 2) * D], values[batch * (B / 2) * D], expanded[batch * (B / 2) * 16 * 8],
                    uint64_t(32), (B / 2) * 16, {1, 1, 0, 4, 4, 1});
        }
        PipeBarrier<PIPE_V>();
        auto result = halves.ReinterpretCast<bfloat16_t>();
        Cast(result, values, RoundMode::CAST_RINT, B * D);
        PipeBarrier<PIPE_V>();
        return result;
    }
};
