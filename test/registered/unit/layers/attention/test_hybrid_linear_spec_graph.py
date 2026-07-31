from types import SimpleNamespace

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-small-1")


def make_backend(*, graph_support: bool):
    return SimpleNamespace(
        needs_cpu_seq_lens=False,
        req_to_token_pool=object(),
        supports_speculative_draft_extend_cuda_graph=graph_support,
        token_to_kv_pool=object(),
    )


class TestHybridLinearSpeculativeGraphCapability(CustomTestCase):
    def test_all_linear_backend_inherits_explicit_graph_support(self):
        full = make_backend(graph_support=False)
        linear = make_backend(graph_support=True)

        backend = HybridLinearAttnBackend(full, linear, full_attn_layers=[])

        self.assertTrue(backend.supports_speculative_draft_extend_cuda_graph)

    def test_hybrid_model_does_not_inherit_linear_only_contract(self):
        full = make_backend(graph_support=False)
        linear = make_backend(graph_support=True)

        backend = HybridLinearAttnBackend(full, linear, full_attn_layers=[0])

        self.assertFalse(backend.supports_speculative_draft_extend_cuda_graph)

    def test_unsupported_linear_backend_remains_eager(self):
        full = make_backend(graph_support=False)
        linear = make_backend(graph_support=False)

        backend = HybridLinearAttnBackend(full, linear, full_attn_layers=[])

        self.assertFalse(backend.supports_speculative_draft_extend_cuda_graph)
