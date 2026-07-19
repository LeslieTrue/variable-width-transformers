"""CPU reference tests for native VWT token-dispatch attention grouping."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml

from attention_grouping import (
    AttentionGroupRouter,
    GroupedSelfAttention,
    attention_group_flop_ratio,
    dense_attention_flops,
    grouped_attention_flops,
    resolve_attention_group_head_schedule,
)
from lm_engine.hf_models.loss import clear_aux_loss, get_aux_loss
from width_varying_config import WidthVaryingConfig
from width_varying_model import WidthVaryingModel


class TestAttentionGroupFlopsTest:
    """Tests for the ideal-uniform FLOP model and integer solver."""

    def test_closed_form_matches_explicit_group_sum(self) -> None:
        """Grouped closed-form FLOPs should equal a slow per-group count."""

        sequence_length = 32
        width = 16
        dense_heads = 4
        group_heads = 2
        groups = 4
        top_k = 2
        tokens_per_group = sequence_length * top_k / groups
        head_dim = width / dense_heads
        per_group = 8 * tokens_per_group * width * group_heads * head_dim
        per_group += 4 * tokens_per_group**2 * group_heads * head_dim
        actual = grouped_attention_flops(
            sequence_length=sequence_length,
            width=width,
            dense_heads=dense_heads,
            group_heads=group_heads,
            num_groups=groups,
            top_k=top_k,
        )
        assert actual == groups * per_group

    def test_solver_matches_dense_span(self) -> None:
        """The integer solver should make the grouped span nearly iso-FLOP."""

        widths = [96, 80, 64]
        dense_heads = [8, 8, 8]
        schedule = resolve_attention_group_head_schedule(
            widths=widths,
            dense_heads=dense_heads,
            sequence_length=128,
            num_groups=8,
            top_k=3,
            compute_match=True,
        )
        ratio = attention_group_flop_ratio(
            widths=widths,
            dense_heads=dense_heads,
            group_heads=schedule,
            sequence_length=128,
            num_groups=8,
            top_k=3,
        )
        assert abs(ratio - 1.0) < 0.02
        assert dense_attention_flops(sequence_length=128, width=96) > 0


class TestAttentionGroupRouterTest:
    """Tests for shared routing, causal compaction, and native aux loss."""

    def test_dispatch_is_stable_causal_and_no_drop(self) -> None:
        """Every selected token should appear once and preserve sequence order."""

        router = AttentionGroupRouter(
            input_width=2,
            num_groups=2,
            top_k=1,
            capacity_multiple=2,
            std=0.1,
        ).eval()
        with torch.no_grad():
            router.gate.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        x = torch.tensor([[[2.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]])
        plan = router(x)
        gathered = []
        for entry in plan.entries:
            valid_indices = entry.gather_indices[0, entry.valid_mask[0]]
            gathered.extend(valid_indices.tolist())
            assert valid_indices.tolist() == sorted(valid_indices.tolist())
            valid_count = int(entry.valid_mask[0].sum())
            assert torch.equal(
                entry.attention_mask[0, 0, :valid_count, :valid_count],
                torch.ones(valid_count, valid_count, dtype=torch.bool).tril(),
            )
        assert sorted(gathered) == [0, 1, 2, 3]

    def test_training_adds_sequence_granular_native_aux_loss(self) -> None:
        """Training should contribute one finite Switch/Z value per sequence."""

        clear_aux_loss()
        router = AttentionGroupRouter(
            input_width=4,
            num_groups=4,
            top_k=2,
            capacity_multiple=2,
            std=0.1,
        ).train()
        plan = router(torch.randn(3, 6, 4))
        aux_loss = get_aux_loss()
        assert isinstance(aux_loss, torch.Tensor)
        assert aux_loss.shape == (3,)
        assert torch.isfinite(aux_loss).all()
        assert plan.assignment_shares.sum().item() == 1.0


class TestGroupedSelfAttentionTest:
    """Reference tests for compact group attention and scattering."""

    def test_compact_forward_matches_slow_per_group_reference(self) -> None:
        """Dispatch/scatter output should equal explicit subsequence attention."""

        torch.manual_seed(7)
        router = AttentionGroupRouter(
            input_width=4,
            num_groups=2,
            top_k=1,
            capacity_multiple=1,
            std=0.1,
        ).eval()
        with torch.no_grad():
            router.gate.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
            )
        layer = GroupedSelfAttention(
            hidden_size=4,
            num_groups=2,
            group_heads=1,
            head_dim=2,
            attention_multiplier=2**-0.5,
            add_bias=False,
            softmax_dropout=0.0,
            dropout=0.0,
            qkv_std=0.1,
            out_std=0.1,
        ).eval()
        x = torch.tensor(
            [
                [
                    [2.0, 0.1, 0.2, 0.3],
                    [-2.0, 0.4, 0.5, 0.6],
                    [1.0, 0.7, 0.8, 0.9],
                    [-1.0, 1.0, 1.1, 1.2],
                ]
            ],
            requires_grad=True,
        )
        plan = router(x)
        rope = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))
        actual = layer(x, rope_cos_sin=rope, dispatch_plan=plan)

        reference = torch.zeros_like(actual)
        for group, entry in zip(layer.groups, plan.entries):
            indices = entry.gather_indices[0, entry.valid_mask[0]]
            subsequence = x[:, indices]
            count = indices.numel()
            group_output = group(
                subsequence,
                rope_cos_sin=(rope[0][:, indices], rope[1][:, indices]),
                attention_mask=torch.ones(1, 1, count, count, dtype=torch.bool).tril(),
                scale=layer.attention_multiplier,
                dropout_p=0.0,
            )
            reference[:, indices] += group_output
        torch.testing.assert_close(actual, reference)

        actual.square().sum().backward()
        assert torch.isfinite(x.grad).all()
        for group in layer.groups:
            assert group.qkv.weight.grad is not None
            assert group.out_proj.weight.grad is not None


class TestWidthVaryingAttentionGroupIntegrationTest:
    """End-to-end tests for shared routing through the VWT model."""

    def test_model_reuses_one_router_across_the_grouped_span(self) -> None:
        """A tiny VWT should run grouped and dense layers with finite gradients."""

        template_path = Path(__file__).parents[1] / "configs" / "dense_200m.yml"
        with template_path.open(encoding="utf-8") as file:
            model_args = yaml.safe_load(file)["model_args"]["pretrained_config"]
        model_args.update(
            hidden_size=64,
            base_width=64,
            bottleneck_ratio=1.0,
            expansion_factor=1.0,
            reduction_factor=1.0,
            max_layer=2,
            num_layers=3,
            quantize_to=16,
            max_position_embeddings=8,
            vocab_size=128,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
            m_width=1,
            m_emb=1,
            layer_norm_epsilon=1e-5,
            attention_num_groups=4,
            attention_num_groups_per_token=2,
            attention_group_num_layers=1,
            attention_group_compute_match=True,
            attention_group_capacity_multiple=2,
        )
        model_args.pop("attention_group_heads_per_layer", None)
        attention = copy.deepcopy(model_args["sequence_mixer_blocks"][0])
        attention["num_attention_heads"] = 4
        attention["num_key_value_heads"] = 4
        attention["attention_multiplier_method"] = None
        model_args["sequence_mixer_blocks"] = [
            copy.deepcopy(attention) for _ in range(3)
        ]
        mlp = copy.deepcopy(model_args["mlp_blocks"][0])
        model_args["mlp_blocks"] = [copy.deepcopy(mlp) for _ in range(3)]

        config = WidthVaryingConfig(**model_args)
        model = WidthVaryingModel(config)
        clear_aux_loss()
        output = model(input_ids=torch.randint(0, 128, (2, 8)), use_cache=False)
        assert output.last_hidden_state.shape == (2, 8, 64)
        assert model.attention_group_router is not None
        blocks = list(model.h.values())
        assert blocks[0].uses_attention_groups
        assert not blocks[1].uses_attention_groups
        metrics = model.get_attention_group_metrics()
        assert metrics["assignment_shares"].shape == (4,)
        assert torch.isfinite(metrics["entropy"])

        output.last_hidden_state.square().mean().backward()
        assert model.attention_group_router.gate.weight.grad is not None
