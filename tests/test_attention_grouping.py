"""CPU reference tests for native VWT token-dispatch attention grouping."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml

from attention_grouping import (
    AttentionGroupMoE,
    AttentionGroupPooledMoE,
    AttentionGroupRouter,
    GroupedSelfAttention,
    MaskedExpertPool,
    attention_group_flop_ratio,
    dense_attention_flops,
    grouped_attention_flops,
    resolve_attention_group_head_schedule,
)
from lm_engine.hf_models.loss import clear_aux_loss, get_aux_loss
from width_varying_config import WidthVaryingConfig
from width_varying_model import WidthVaryingModel


def _make_masked_pool(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int = 2,
    experts_per_token: int = 1,
    auxiliary_loss_scale: float = 1.0,
) -> MaskedExpertPool:
    """Build a small bias-free SwiGLU pool for CPU reference tests.

    Args:
        hidden_size (int): Input and output width.
        intermediate_size (int): Per-expert SwiGLU width.
        num_experts (int): Stored experts.
        experts_per_token (int): Selected experts per token.
        auxiliary_loss_scale (float): Router auxiliary-loss multiplier.

    Returns:
        MaskedExpertPool: Initialized native VWT expert pool.
    """

    return MaskedExpertPool(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        shared_intermediate_size=None,
        use_interleaved_weights_for_shared_experts=False,
        use_interleaved_weights=False,
        shared_expert_gating=False,
        normalized_topk=True,
        num_experts=num_experts,
        num_experts_per_tok=experts_per_token,
        add_bias=False,
        activation_function="swiglu",
        dropout=0.0,
        init_method="mup",
        initializer_range=0.1,
        m_width=1,
        num_layers=2,
        use_depth_scaled_init=True,
        use_padding_free_transformer=False,
        sequence_parallel=False,
        auxiliary_loss_scale=auxiliary_loss_scale,
    )


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


class TestAttentionGroupMoETest:
    """Reference tests for the shared-plus-column-private expert MLP."""

    def test_sparse_private_dispatch_matches_dense_reference(self) -> None:
        """Shared-once and per-column private updates should match a slow sum."""

        torch.manual_seed(11)
        router = AttentionGroupRouter(
            input_width=4,
            num_groups=2,
            top_k=2,
            capacity_multiple=1,
            std=0.1,
        ).eval()
        shared = torch.nn.Linear(4, 4, bias=False)
        module = AttentionGroupMoE(
            hidden_size=4,
            num_private_experts=2,
            shared_expert=shared,
            private_expert_factory=lambda: torch.nn.Linear(4, 4, bias=False),
            router_std=0.1,
        ).eval()
        x = torch.randn(2, 3, 4, requires_grad=True)
        plan = router(x)
        actual = module(x, plan)

        expert_gates = torch.softmax(module.expert_gate(x).float(), dim=-1).type_as(x)
        reference = module.shared_expert(x) * expert_gates[..., :1]
        for expert, entry in zip(module.private_experts, plan.entries):
            for batch_index in range(x.shape[0]):
                valid = entry.valid_mask[batch_index]
                indices = entry.gather_indices[batch_index, valid]
                weights = entry.gates[batch_index, valid]
                weights = weights * expert_gates[batch_index, indices, 1]
                reference[batch_index, indices] += (
                    expert(x[batch_index, indices]) * weights.unsqueeze(-1)
                )
        torch.testing.assert_close(actual, reference)

        actual.square().sum().backward()
        assert module.shared_expert.weight.grad is not None
        assert module.expert_gate.weight.grad is not None
        for expert in module.private_experts:
            assert expert.weight.grad is not None

    def test_single_column_runs_shared_and_private_once(self) -> None:
        """Dense later blocks should need no attention dispatch plan."""

        module = AttentionGroupMoE(
            hidden_size=3,
            num_private_experts=1,
            shared_expert=torch.nn.Linear(3, 3, bias=False),
            private_expert_factory=lambda: torch.nn.Linear(3, 3, bias=False),
            router_std=0.1,
        )
        x = torch.randn(2, 5, 3)
        gates = torch.softmax(module.expert_gate(x).float(), dim=-1).type_as(x)
        expected = module.shared_expert(x) * gates[..., :1]
        expected += module.private_experts[0](x) * gates[..., 1:]
        torch.testing.assert_close(module(x), expected)


class TestMaskedExpertPoolTest:
    """Reference tests for valid-token expert-pool dispatch."""

    def test_masked_padding_matches_unpadded_pool(self) -> None:
        """Capacity-padding rows should neither change nor receive outputs."""

        torch.manual_seed(13)
        pool = _make_masked_pool(hidden_size=4, intermediate_size=8).eval()
        valid_x = torch.randn(1, 3, 4)
        padded_x = torch.cat([valid_x, torch.randn(1, 2, 4)], dim=1)
        valid_mask = torch.tensor([[True, True, True, False, False]])

        expected = pool(valid_x)
        actual = pool(padded_x, valid_mask=valid_mask)

        torch.testing.assert_close(actual[:, :3], expected)
        assert torch.count_nonzero(actual[:, 3:]) == 0

    def test_masked_training_adds_sequence_granular_aux_loss(self) -> None:
        """Sparse pool balancing should ignore padding and remain per sequence."""

        torch.manual_seed(17)
        clear_aux_loss()
        pool = _make_masked_pool(
            hidden_size=4,
            intermediate_size=8,
            num_experts=3,
            experts_per_token=2,
            auxiliary_loss_scale=0.5,
        ).train()
        x = torch.randn(2, 4, 4, requires_grad=True)
        valid_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, False]]
        )

        output = pool(x, valid_mask=valid_mask)
        auxiliary_loss = get_aux_loss()

        assert auxiliary_loss.shape == (2,)
        assert torch.isfinite(auxiliary_loss).all()
        assert torch.count_nonzero(output[~valid_mask]) == 0
        output.square().sum().backward()
        assert pool.gate.weight.grad is not None
        assert pool.c_fc.weight.grad is not None
        assert pool.c_proj.weight.grad is not None


class TestAttentionGroupPooledMoETest:
    """Reference tests for public and group-private expert pools."""

    def test_sparse_pools_match_explicit_group_sum(self) -> None:
        """Pooled dispatch should match a slow unpadded group reference."""

        torch.manual_seed(19)
        router = AttentionGroupRouter(
            input_width=4,
            num_groups=2,
            top_k=2,
            capacity_multiple=1,
            std=0.1,
        ).eval()
        module = AttentionGroupPooledMoE(
            hidden_size=4,
            num_private_groups=2,
            public_pool=_make_masked_pool(
                hidden_size=4, intermediate_size=8
            ),
            private_pool_factory=lambda: _make_masked_pool(
                hidden_size=4,
                intermediate_size=4,
                auxiliary_loss_scale=0.5,
            ),
            router_std=0.1,
        ).eval()
        x = torch.randn(1, 4, 4, requires_grad=True)
        plan = router(x)

        actual = module(x, plan)
        path_gates = torch.softmax(
            module.public_private_gate(x).float(), dim=-1
        ).type_as(x)
        reference = module.public_pool(x) * path_gates[..., :1]
        for pool, entry in zip(module.private_pools, plan.entries):
            valid = entry.valid_mask[0]
            indices = entry.gather_indices[0, valid]
            weights = entry.gates[0, valid] * path_gates[0, indices, 1]
            reference[0, indices] += (
                pool(x[:, indices])[0] * weights.unsqueeze(-1)
            )

        torch.testing.assert_close(actual, reference)
        actual.square().sum().backward()
        assert module.public_private_gate.weight.grad is not None
        assert module.public_pool.c_fc.weight.grad is not None
        assert all(pool.c_fc.weight.grad is not None for pool in module.private_pools)


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

    def test_model_refreshes_routing_at_each_grouped_depth_block(self) -> None:
        """Each routed block should own and backpropagate through its router."""

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
            attention_group_depths=[1, 1, 1],
            attention_num_groups_by_block=[4, 2, 1],
            attention_num_groups_per_token_by_block=[2, 1, 1],
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
        blocks = list(model.h.values())
        assert [block.attention_group_num_groups for block in blocks] == [4, 2, 1]
        assert [block.uses_attention_groups for block in blocks] == [True, True, False]
        assert list(model.attention_group_routers) == ["0", "1"]

        clear_aux_loss()
        output = model(input_ids=torch.randint(0, 128, (2, 8)), use_cache=False)
        metrics = model.get_attention_group_metrics()
        assert metrics["block_0/assignment_shares"].shape == (4,)
        assert metrics["block_1/assignment_shares"].shape == (2,)
        output.last_hidden_state.square().mean().backward()
        assert all(
            router.gate.weight.grad is not None
            for router in model.attention_group_routers.values()
        )

    def test_model_builds_three_block_sp50_experts(self) -> None:
        """SP-50 should split dense MLP compute equally across both paths."""

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
            attention_group_depths=[1, 1, 1],
            attention_num_groups_by_block=[4, 2, 1],
            attention_num_groups_per_token_by_block=[2, 1, 1],
            attention_group_compute_match=True,
            attention_group_capacity_multiple=2,
            attention_group_moe=True,
            attention_group_moe_shared_expansion_ratio=2.0,
            attention_group_moe_private_active_expansion_ratio=2.0,
            attention_group_moe_intermediate_multiple=8,
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
        blocks = list(model.h.values())
        assert all(isinstance(block.mlp_block, AttentionGroupMoE) for block in blocks)
        assert [block.mlp_block.num_private_experts for block in blocks] == [4, 2, 1]
        assert [block.mlp_block.shared_expert.c_proj.in_features for block in blocks] == [
            128,
            128,
            128,
        ]
        assert [
            block.mlp_block.private_experts[0].c_proj.in_features for block in blocks
        ] == [64, 128, 128]
        assert config.attention_group_moe_shared_intermediate_sizes == [128] * 3
        assert config.attention_group_moe_private_intermediate_sizes == [64, 128, 128]

        clear_aux_loss()
        output = model(input_ids=torch.randint(0, 128, (2, 8)), use_cache=False)
        output.last_hidden_state.square().mean().backward()
        assert all(block.mlp_block.expert_gate.weight.grad is not None for block in blocks)

    def test_model_builds_prefix_public_private_expert_pools(self) -> None:
        """Only grouped prefix layers should replace their dense MLP."""

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
            attention_group_pooled_moe=True,
            attention_group_public_experts=4,
            attention_group_public_experts_per_token=1,
            attention_group_private_experts_per_group=3,
            attention_group_private_experts_per_token=1,
            attention_group_pooled_moe_public_active_expansion_ratio=2.0,
            attention_group_pooled_moe_private_active_expansion_ratio=2.0,
            attention_group_pooled_moe_intermediate_multiple=8,
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
        blocks = list(model.h.values())

        assert isinstance(blocks[0].mlp_block, AttentionGroupPooledMoE)
        assert not isinstance(blocks[1].mlp_block, AttentionGroupPooledMoE)
        assert not isinstance(blocks[2].mlp_block, AttentionGroupPooledMoE)
        assert blocks[0].mlp_block.public_pool.num_experts == 4
        assert len(blocks[0].mlp_block.private_pools) == 4
        assert all(
            pool.num_experts == 3
            for pool in blocks[0].mlp_block.private_pools
        )
        assert config.attention_group_pooled_moe_public_intermediate_sizes == [
            128,
            0,
            0,
        ]
        assert config.attention_group_pooled_moe_private_intermediate_sizes == [
            64,
            0,
            0,
        ]

        clear_aux_loss()
        output = model(input_ids=torch.randint(0, 128, (2, 8)), use_cache=False)
        auxiliary_loss = get_aux_loss()
        assert auxiliary_loss.shape == (2,)
        assert torch.isfinite(auxiliary_loss).all()
        output.last_hidden_state.square().mean().backward()
        assert blocks[0].mlp_block.public_pool.gate.weight.grad is not None
        assert all(
            pool.gate.weight.grad is not None
            for pool in blocks[0].mlp_block.private_pools
        )
