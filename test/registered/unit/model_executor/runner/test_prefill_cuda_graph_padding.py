import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.enable_lora = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    def test_full_graph_metadata_uses_static_extend_start_loc(self):
        captured = {}

        class Backend:
            def init_forward_metadata_out_graph(self, forward_batch):
                captured["forward_batch"] = forward_batch

        runner = self._make_runner()
        runner._is_full_backend = True
        runner._capture_req_slots = 4
        runner._full_cg_seq_lens_cpu = torch.zeros(4, dtype=torch.int64)
        runner._prefill_static_buffers = {
            "seq_lens": torch.tensor([8, 4, 0, 0]),
            "req_pool_indices": torch.tensor([3, 7, 0, 0]),
            "extend_seq_lens": torch.tensor([3, 2, 0, 0]),
            "extend_prefix_lens": torch.tensor([5, 2, 0, 0]),
            "extend_start_loc": torch.tensor([0, 3, 5, 5]),
        }
        runner.model_runner = SimpleNamespace(attn_backend=Backend())
        forward_batch = SimpleNamespace(
            batch_size=2,
            seq_lens_cpu=torch.tensor([8, 4], dtype=torch.int64),
        )

        runner._prepare_forward_metadata_for_replay(
            forward_batch, static_forward_batch=SimpleNamespace(), num_tokens=16
        )

        padded = captured["forward_batch"]
        self.assertEqual(padded.batch_size, 4)
        self.assertEqual(padded.extend_start_loc.tolist(), [0, 3, 5, 5])
        self.assertEqual(
            padded.extend_start_loc.data_ptr(),
            runner._prefill_static_buffers["extend_start_loc"].data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
