"""Numerical parity tests for the native RWKV-7 Triton kernels."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.rwkv7_kernels import wkv_recurrent
from sglang.srt.layers.attention.rwkv7_kernels.fused import (
    fused_gate_corr,
    fused_kk_kmix,
    fused_lerp6,
)
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


if __name__ == "__main__":
    unittest.main()
