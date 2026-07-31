"""Numerical parity tests for the native RWKV-7 Triton kernels."""

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.rwkv7 import Rwkv7Config
from sglang.srt.layers.attention.rwkv7_kernels import (
    layernorm_token_shift_lerp1_decode,
    layernorm_token_shift_lerp6_decode,
    token_shift_lerp6_decode,
    token_shift_lerp6_packed_varlen,
    token_shift_packed_varlen,
    wkv_recurrent,
)
from sglang.srt.layers.attention.rwkv7_kernels.fused import (
    can_fuse_lowrank_controls,
    fused_gate_corr,
    fused_groupnorm_gate_corr,
    fused_kk_kmix,
    fused_lerp6,
    fused_lowrank_controls,
    is_profitable_fused_lowrank_shape,
)
from sglang.srt.models.rwkv7 import Rwkv7Attention
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")


def _reference_sequence(r, w, k, v, kk, a, state, scale=1.0):
    """Straight-line published RWKV-7 recurrence in fp32 state precision."""
    outputs = []
    state = state.float().clone()
    for t in range(r.shape[0]):
        rt = r[t].float() * scale
        wt = w[t].float()
        kt = k[t].float()
        vt = v[t].float()
        kkt = kk[t].float()
        at = a[t].float()

        sa = ((-kkt).unsqueeze(-1) * state).sum(dim=-2)
        state = (
            wt.exp().unsqueeze(-1) * state
            + (kkt * at).unsqueeze(-1) * sa.unsqueeze(-2)
            + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        )
        outputs.append((state * rt.unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outputs).to(v.dtype), state


@unittest.skipUnless(torch.cuda.is_available(), "RWKV-7 Triton kernels require CUDA")
class TestRwkv7Kernels(unittest.TestCase):
    dtype = torch.bfloat16
    heads = 2
    head_dim = 64

    def setUp(self):
        torch.manual_seed(0x7A11)

    def _inputs(self, batch, length):
        shape = (batch, length, self.heads, self.head_dim)
        r = torch.randn(shape, device="cuda", dtype=self.dtype) * 0.2
        w = -torch.rand(shape, device="cuda", dtype=self.dtype) * 1.5
        k = torch.randn(shape, device="cuda", dtype=self.dtype) * 0.2
        v = torch.randn(shape, device="cuda", dtype=self.dtype) * 0.2
        kk = F.normalize(torch.randn(shape, device="cuda", dtype=self.dtype), dim=-1)
        a = torch.sigmoid(torch.randn(shape, device="cuda", dtype=self.dtype))
        return r, w, k, v, kk, a

    def test_fixed_batch_matches_published_recurrence(self):
        batch, length = 4, 5
        inputs = self._inputs(batch, length)
        initial = (
            torch.randn(
                batch,
                self.heads,
                self.head_dim,
                self.head_dim,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.02
        )

        actual_o, actual_state = wkv_recurrent(
            *inputs, initial_state=initial, output_final_state=True
        )
        expected = [
            _reference_sequence(*(x[b] for x in inputs), initial[b])
            for b in range(batch)
        ]
        expected_o = torch.stack([item[0] for item in expected])
        expected_state = torch.stack([item[1] for item in expected])

        torch.testing.assert_close(actual_o, expected_o, rtol=0.02, atol=0.02)
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-5)

    def test_packed_varlen_matches_independent_sequences(self):
        lengths = [1, 3, 7]
        total = sum(lengths)
        inputs = self._inputs(1, total)
        initial = (
            torch.randn(
                len(lengths),
                self.heads,
                self.head_dim,
                self.head_dim,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.02
        )
        cu_seqlens = torch.tensor([0, 1, 4, total], device="cuda", dtype=torch.int64)

        with patch.dict(os.environ, {"SGLANG_RWKV7_FAST_FP16_PREFILL": "1"}):
            actual_o, actual_state = wkv_recurrent(
                *inputs,
                initial_state=initial,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            )

        expected_outputs = []
        expected_states = []
        start = 0
        for index, length in enumerate(lengths):
            result = _reference_sequence(
                *(x[0, start : start + length] for x in inputs), initial[index]
            )
            expected_outputs.append(result[0])
            expected_states.append(result[1])
            start += length

        expected_o = torch.cat(expected_outputs).unsqueeze(0)
        expected_state = torch.stack(expected_states)
        torch.testing.assert_close(actual_o, expected_o, rtol=0.02, atol=0.02)
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-5)

    def test_fp16_indexed_varlen_state_matches_published_recurrence(self):
        lengths = [3, 0, 5]
        total = sum(lengths)
        scale = self.head_dim**-0.5
        inputs = tuple(tensor.to(torch.float16) for tensor in self._inputs(1, total))
        state_pool = (
            torch.randn(
                7,
                self.heads,
                self.head_dim,
                self.head_dim,
                device="cuda",
                dtype=torch.float16,
            )
            * 0.02
        )
        original_pool = state_pool.clone()
        cache_indices = torch.tensor([2, -1, 5], device="cuda", dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 3, 3, total], device="cuda", dtype=torch.int32)

        actual_o, returned_state = wkv_recurrent(
            *inputs,
            scale=scale,
            state_pool=state_pool,
            cache_indices=cache_indices,
            cu_seqlens=cu_seqlens,
        )
        self.assertIsNone(returned_state)

        expected_outputs = []
        start = 0
        for length, slot in ((3, 2), (5, 5)):
            expected_o, expected_state = _reference_sequence(
                *(x[0, start : start + length] for x in inputs),
                original_pool[slot],
                scale=scale,
            )
            expected_outputs.append(expected_o)
            torch.testing.assert_close(
                state_pool[slot].float(), expected_state, rtol=0.01, atol=0.002
            )
            start += length

        torch.testing.assert_close(
            actual_o,
            torch.cat(expected_outputs).unsqueeze(0),
            rtol=0.03,
            atol=0.001,
        )
        untouched = [index for index in range(len(state_pool)) if index not in (2, 5)]
        torch.testing.assert_close(state_pool[untouched], original_pool[untouched])

    def test_fp16_fast_prefill_matches_published_recurrence(self):
        lengths = [3, 5]
        total = sum(lengths)
        inputs = tuple(tensor.to(torch.float16) for tensor in self._inputs(1, total))
        initial = (
            torch.randn(
                len(lengths),
                self.heads,
                self.head_dim,
                self.head_dim,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.02
        )
        cu_seqlens = torch.tensor(
            [0, lengths[0], total], device="cuda", dtype=torch.int32
        )

        actual_o, actual_state = wkv_recurrent(
            *inputs,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        expected = []
        start = 0
        for index, length in enumerate(lengths):
            expected.append(
                _reference_sequence(
                    *(x[0, start : start + length] for x in inputs), initial[index]
                )
            )
            start += length
        expected_o = torch.cat([item[0] for item in expected]).unsqueeze(0)
        expected_state = torch.stack([item[1] for item in expected])
        torch.testing.assert_close(actual_o, expected_o, rtol=0.02, atol=0.02)
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-5)

    def test_indexed_decode_updates_only_live_slots(self):
        inputs = self._inputs(3, 1)
        state_pool = (
            torch.randn(
                8,
                self.heads,
                self.head_dim,
                self.head_dim,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.02
        )
        original_pool = state_pool.clone()
        cache_indices = torch.tensor([2, 5, -1], device="cuda", dtype=torch.int32)

        actual_o, returned_state = wkv_recurrent(
            *inputs, state_pool=state_pool, cache_indices=cache_indices
        )
        self.assertIsNone(returned_state)

        for batch_index, slot in enumerate((2, 5)):
            expected_o, expected_state = _reference_sequence(
                *(x[batch_index] for x in inputs), original_pool[slot]
            )
            torch.testing.assert_close(
                actual_o[batch_index], expected_o, rtol=0.02, atol=0.02
            )
            torch.testing.assert_close(
                state_pool[slot], expected_state, rtol=2e-5, atol=2e-5
            )

        untouched = [index for index in range(len(state_pool)) if index not in (2, 5)]
        torch.testing.assert_close(state_pool[untouched], original_pool[untouched])

    def test_indexed_varlen_ignores_zero_length_graph_slots(self):
        lengths = [3, 0, 2, 0]
        inputs = self._inputs(1, sum(lengths))
        state_pool = (
            torch.randn(
                8,
                self.heads,
                self.head_dim,
                self.head_dim,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.02
        )
        original_pool = state_pool.clone()
        cache_indices = torch.tensor([2, -1, 5, -1], device="cuda", dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 3, 3, 5, 5], device="cuda", dtype=torch.int32)

        actual_o, returned_state = wkv_recurrent(
            *inputs,
            state_pool=state_pool,
            cache_indices=cache_indices,
            cu_seqlens=cu_seqlens,
        )
        self.assertIsNone(returned_state)

        expected_outputs = []
        start = 0
        for length, slot in ((3, 2), (2, 5)):
            expected_o, expected_state = _reference_sequence(
                *(x[0, start : start + length] for x in inputs), original_pool[slot]
            )
            expected_outputs.append(expected_o)
            torch.testing.assert_close(
                state_pool[slot], expected_state, rtol=2e-5, atol=2e-5
            )
            start += length

        torch.testing.assert_close(
            actual_o,
            torch.cat(expected_outputs).unsqueeze(0),
            rtol=0.02,
            atol=0.02,
        )
        untouched = [index for index in range(len(state_pool)) if index not in (2, 5)]
        torch.testing.assert_close(state_pool[untouched], original_pool[untouched])

    def test_token_shift_masks_zero_length_graph_slots(self):
        token_count, hidden = 5, 128
        x = torch.randn(token_count, hidden, device="cuda", dtype=self.dtype)
        conv = torch.randn(8, hidden, 1, device="cuda", dtype=torch.float32)
        original_conv = conv.clone()
        query_start_loc = torch.tensor(
            [0, 3, 3, 5, 5], device="cuda", dtype=torch.int32
        )
        cache_indices = torch.tensor([2, -1, 5, -1], device="cuda", dtype=torch.int32)

        actual = token_shift_packed_varlen(x, conv, query_start_loc, cache_indices)
        expected = torch.empty_like(x)
        expected[1:] = x[:-1]
        expected[0] = original_conv[2, :, 0].to(x.dtype)
        expected[3] = original_conv[5, :, 0].to(x.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(conv[2, :, 0], x[2].float(), rtol=0, atol=0)
        torch.testing.assert_close(conv[5, :, 0], x[4].float(), rtol=0, atol=0)
        untouched = [index for index in range(len(conv)) if index not in (2, 5)]
        torch.testing.assert_close(conv[untouched], original_conv[untouched])

    def test_fused_token_shift_lerp6_packed_is_bit_exact(self):
        token_count, hidden = 5, 128
        x = torch.randn(token_count, hidden, device="cuda", dtype=self.dtype)
        mix = torch.randn(6, hidden, device="cuda", dtype=self.dtype)
        conv = torch.randn(8, hidden, 1, device="cuda", dtype=torch.float32)
        reference_conv = conv.clone()
        query_start_loc = torch.tensor(
            [0, 3, 3, 5, 5], device="cuda", dtype=torch.int32
        )
        cache_indices = torch.tensor([2, -1, 5, -1], device="cuda", dtype=torch.int32)

        shifted = token_shift_packed_varlen(
            x, reference_conv, query_start_loc, cache_indices
        )
        expected = fused_lerp6(x, shifted, mix)
        actual = token_shift_lerp6_packed_varlen(
            x, conv, mix, query_start_loc, cache_indices
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(conv, reference_conv, rtol=0, atol=0)

    def test_fused_token_shift_lerp6_decode_is_bit_exact(self):
        batch, hidden = 4, 128
        x = torch.randn(batch, hidden, device="cuda", dtype=self.dtype)
        mix = torch.randn(6, hidden, device="cuda", dtype=self.dtype)
        conv = torch.randn(8, hidden, 1, device="cuda", dtype=torch.float32)
        reference_conv = conv.clone()
        cache_indices = torch.tensor([2, 3, 5, 6], device="cuda", dtype=torch.int32)
        shifted = reference_conv[cache_indices, :, 0].to(x.dtype)
        reference_conv[cache_indices, :, 0] = x.to(reference_conv.dtype)
        expected = fused_lerp6(x, shifted, mix)

        actual = token_shift_lerp6_decode(x, conv, mix, cache_indices)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(conv, reference_conv, rtol=0, atol=0)

    def test_speculative_snapshots_do_not_advance_persistent_state(self):
        batch, length = 3, 4
        packed_inputs = self._inputs(1, batch * length)
        state_pool = (
            torch.randn(
                9,
                self.heads,
                self.head_dim,
                self.head_dim,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.02
        )
        original_pool = state_pool.clone()
        cache_indices = torch.tensor([2, 5, 7], device="cuda", dtype=torch.int32)
        scratch_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
        scratch = torch.full(
            (batch + 1, length, self.heads, self.head_dim, self.head_dim),
            float("nan"),
            device="cuda",
            dtype=torch.float32,
        )
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * length,
            length,
            device="cuda",
            dtype=torch.int64,
        )

        actual_o, returned_state = wkv_recurrent(
            *packed_inputs,
            state_pool=state_pool,
            cache_indices=cache_indices,
            update_state_pool=False,
            intermediate_state=scratch,
            intermediate_state_indices=scratch_indices,
            cu_seqlens=cu_seqlens,
        )
        self.assertIsNone(returned_state)
        torch.testing.assert_close(state_pool, original_pool)

        expected_outputs = []
        for batch_index, slot in enumerate(cache_indices.tolist()):
            start = batch_index * length
            state = original_pool[slot]
            for step in range(length):
                expected_o, state = _reference_sequence(
                    *(x[0, start + step : start + step + 1] for x in packed_inputs),
                    state,
                )
                expected_outputs.append(expected_o)
                torch.testing.assert_close(
                    scratch[batch_index, step], state, rtol=2e-5, atol=2e-5
                )

        expected_o = torch.cat(expected_outputs).unsqueeze(0)
        torch.testing.assert_close(actual_o, expected_o, rtol=0.02, atol=0.02)

    def test_fused_elementwise_ops_are_bit_exact(self):
        tokens, hidden, heads = 5, 128, 2
        x = torch.randn(tokens, hidden, device="cuda", dtype=self.dtype)
        shifted = torch.randn_like(x)
        mix = torch.randn(6, hidden, device="cuda", dtype=self.dtype)
        torch.testing.assert_close(
            fused_lerp6(x, shifted, mix),
            x.unsqueeze(0) + mix.unsqueeze(1) * (shifted - x).unsqueeze(0),
            rtol=0,
            atol=0,
        )

        k = torch.randn_like(x)
        a = torch.sigmoid(torch.randn_like(x))
        k_k = torch.randn(hidden, device="cuda", dtype=self.dtype)
        k_a = torch.randn(hidden, device="cuda", dtype=self.dtype)
        actual_kk, actual_k = fused_kk_kmix(k, a, k_k, k_a, heads)
        expected_kk = (k * k_k).view(tokens, heads, -1)
        expected_kk = expected_kk / expected_kk.norm(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        expected_k = k + k * (a - 1.0) * k_a
        torch.testing.assert_close(actual_kk, expected_kk, rtol=0, atol=0)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)

        o_norm = torch.randn_like(x)
        r = torch.randn(tokens, heads, hidden // heads, device="cuda", dtype=self.dtype)
        k_head = k.view(tokens, heads, -1)
        v = torch.randn_like(r)
        r_k = torch.randn(heads, hidden // heads, device="cuda", dtype=self.dtype)
        g = torch.sigmoid(torch.randn_like(x))
        expected = o_norm + ((r * k_head * r_k).sum(-1, keepdim=True) * v).view(
            tokens, hidden
        )
        expected = expected * g
        actual = fused_gate_corr(o_norm, r, k_head, r_k, v, g, heads)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        norm = torch.nn.GroupNorm(heads, hidden, eps=64e-5, device="cuda").to(
            self.dtype
        )
        raw_o = torch.randn_like(x)
        expected_norm = norm(raw_o)
        expected = expected_norm + ((r * k_head * r_k).sum(-1, keepdim=True) * v).view(
            tokens, hidden
        )
        expected = expected * g
        actual = fused_groupnorm_gate_corr(
            raw_o,
            r,
            k_head,
            r_k,
            v,
            g,
            norm.weight,
            norm.bias,
            heads,
            norm.eps,
        )
        # GroupNorm's reduction tree differs from ATen's implementation. The
        # fused kernel remains within the same low-precision tolerance used by
        # the recurrent kernel and is covered by the end-to-end logit gate.
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_fused_lowrank_controls_match_canonical_modules(self):
        class LowRank(nn.Module):
            def __init__(self, hidden, rank, activation, bias):
                super().__init__()
                self.lora = nn.Sequential(
                    nn.Linear(hidden, rank, bias=False),
                    activation,
                    nn.Linear(rank, hidden, bias=bias),
                ).to(device="cuda", dtype=torch.float16)

            def forward(self, value):
                return self.lora(value)

        hidden = 256
        ranks = (24, 24, 64, 16)
        w_lora = LowRank(hidden, ranks[0], nn.Tanh(), True)
        a_lora = LowRank(hidden, ranks[1], nn.Identity(), True)
        g_lora = LowRank(hidden, ranks[2], nn.Sigmoid(), False)
        v_lora = LowRank(hidden, ranks[3], nn.Identity(), True)
        self.assertTrue(can_fuse_lowrank_controls(w_lora, a_lora, g_lora, v_lora))

        for batch in (1, 8):
            inputs = [
                torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
                for _ in range(4)
            ]
            value = torch.randn_like(inputs[0])
            v_first = torch.randn_like(value)
            expected_w = -torch.sigmoid(w_lora(inputs[0])) * (2.718281828459045**-0.5)
            expected_a = torch.sigmoid(a_lora(inputs[1]))
            expected_g = g_lora(inputs[2])
            expected_v = value + (v_first - value) * torch.sigmoid(v_lora(inputs[3]))
            actual = fused_lowrank_controls(
                *inputs,
                value,
                v_first,
                w_lora,
                a_lora,
                g_lora,
                v_lora,
            )
            for result, expected in zip(
                actual, (expected_w, expected_a, expected_g, expected_v)
            ):
                torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)

        actual_w, actual_a, actual_g, actual_v = fused_lowrank_controls(
            *inputs,
            value,
            v_first,
            w_lora,
            a_lora,
            g_lora,
        )
        torch.testing.assert_close(actual_v, value, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_w,
            -torch.sigmoid(w_lora(inputs[0])) * (2.718281828459045**-0.5),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            actual_a, torch.sigmoid(a_lora(inputs[1])), rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(actual_g, g_lora(inputs[2]), rtol=2e-2, atol=2e-2)

    def test_fused_lowrank_dispatch_policy(self):
        for batch in (1, 2, 4, 8):
            self.assertTrue(is_profitable_fused_lowrank_shape(2048, batch))
            self.assertTrue(is_profitable_fused_lowrank_shape(2560, batch))
        self.assertTrue(is_profitable_fused_lowrank_shape(4096, 4))
        self.assertFalse(is_profitable_fused_lowrank_shape(4096, 8))
        self.assertFalse(is_profitable_fused_lowrank_shape(5120, 1))
        self.assertFalse(is_profitable_fused_lowrank_shape(2048, 16))

    def test_fused_lowrank_tensorcore_up_matches_modules(self):
        class LowRank(nn.Module):
            def __init__(self, hidden, rank, activation, bias):
                super().__init__()
                self.lora = nn.Sequential(
                    nn.Linear(hidden, rank, bias=False),
                    activation,
                    nn.Linear(rank, hidden, bias=bias),
                ).to(device="cuda", dtype=torch.float16)

            def forward(self, value):
                return self.lora(value)

        batch, hidden = 8, 1024
        modules = (
            LowRank(hidden, 48, nn.Tanh(), True),
            LowRank(hidden, 48, nn.Identity(), True),
            LowRank(hidden, 128, nn.Sigmoid(), False),
            LowRank(hidden, 32, nn.Identity(), True),
        )
        inputs = [
            torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
            for _ in range(4)
        ]
        value = torch.randn_like(inputs[0])
        v_first = torch.randn_like(value)
        w_lora, a_lora, g_lora, v_lora = modules
        expected = (
            -torch.sigmoid(w_lora(inputs[0])) * (2.718281828459045**-0.5),
            torch.sigmoid(a_lora(inputs[1])),
            g_lora(inputs[2]),
            value + (v_first - value) * torch.sigmoid(v_lora(inputs[3])),
        )
        actual = fused_lowrank_controls(
            *inputs,
            value,
            v_first,
            w_lora,
            a_lora,
            g_lora,
            v_lora,
        )
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-2)

    def test_layernorm_token_shift_mix_decode_matches_torch(self):
        batch, hidden, slots = 8, 256, 16
        x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
        conv = torch.randn(slots, hidden, 1, device="cuda", dtype=torch.float32)
        mix6 = torch.randn(6, hidden, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden, device="cuda", dtype=torch.float16)
        indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)

        normalized = F.layer_norm(x, (hidden,), weight, bias, 1e-5)
        shifted = conv[indices.long(), :, 0].to(x.dtype)
        delta = shifted - normalized
        expected6 = torch.stack([normalized + mix6[i] * delta for i in range(6)], dim=0)
        conv6 = conv.clone()
        actual6 = layernorm_token_shift_lerp6_decode(
            x, conv6, mix6, weight, bias, 1e-5, indices
        )
        torch.testing.assert_close(actual6, expected6, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            conv6[indices.long(), :, 0], normalized.float(), rtol=2e-2, atol=2e-2
        )

        mix1 = mix6[0]
        expected1 = normalized + mix1 * delta
        conv1 = conv.clone()
        actual1 = layernorm_token_shift_lerp1_decode(
            x, conv1, mix1, weight, bias, 1e-5, indices
        )
        torch.testing.assert_close(actual1, expected1, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            conv1[indices.long(), :, 0], normalized.float(), rtol=2e-2, atol=2e-2
        )

    def test_rkv_stacked_storage_survives_dtype_apply(self):
        config = Rwkv7Config(
            hidden_size=2048,
            num_hidden_layers=2,
            head_dim=64,
            num_heads=32,
            intermediate_size=4096,
        )
        with torch.device("cuda"):
            attention = Rwkv7Attention(config, layer_id=0)
        attention = attention.to(dtype=torch.float16)

        stacked = attention._rkv_stacked_weight
        modules = (attention.r_proj, attention.k_proj, attention.v_proj)
        self.assertIsNotNone(stacked)
        for index, module in enumerate(modules):
            self.assertEqual(module.weight.data_ptr(), stacked[index].data_ptr())
            self.assertTrue(hasattr(module.weight, "weight_loader"))
        self.assertTrue(
            {"r_proj.weight", "k_proj.weight", "v_proj.weight"}.issubset(
                attention.state_dict()
            )
        )

        attention.k_proj.weight.data.fill_(3)
        self.assertEqual(float(stacked[1, 0, 0]), 3.0)

    def test_rkv_stacked_projection_matches_canonical_linears(self):
        config = Rwkv7Config(
            hidden_size=2048,
            num_hidden_layers=2,
            head_dim=64,
            num_heads=32,
            intermediate_size=4096,
        )
        with torch.device("cuda"):
            attention = Rwkv7Attention(config, layer_id=0)
        attention = attention.to(dtype=torch.float16)

        stacked = attention._rkv_stacked_weight
        self.assertIsNotNone(stacked)
        modules = (attention.r_proj, attention.k_proj, attention.v_proj)
        for tokens in (8, 256):
            mixed = torch.randn(
                3, tokens, config.hidden_size, device="cuda", dtype=torch.float16
            )
            actual = torch.bmm(mixed, stacked.transpose(1, 2))
            expected = torch.stack(
                [module(mixed[index])[0] for index, module in enumerate(modules)]
            )
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
