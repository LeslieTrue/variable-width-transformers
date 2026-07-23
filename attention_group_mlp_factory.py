"""Factories for shared/private and pooled attention-group MLP blocks."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from lm_engine.hf_models.modeling_utils.mlp_blocks import get_mlp_block

from attention_grouping import AttentionGroupMoE, AttentionGroupPooledMoE
from masked_expert_pool import MaskedExpertPool


def build_attention_group_mlp(
    *,
    config: Any,
    layer_idx: int,
    hidden_size: int,
    num_groups: int,
    uses_attention_groups: bool,
    dense_expert: nn.Module,
    layer_initializer_range: float,
    original_hidden_size: int,
    original_initializer_range: float,
    use_padding_free_transformer: bool,
    sequence_parallel: bool,
) -> nn.Module:
    """Replace a native dense MLP with the requested attention-group variant.

    Args:
        config (Any): Mutable VWT configuration.
        layer_idx (int): Zero-based transformer layer index.
        hidden_size (int): Layer residual width.
        num_groups (int): Attention groups at this layer.
        uses_attention_groups (bool): Whether the layer belongs to a routed
            attention prefix/block.
        dense_expert (nn.Module): Native dense SwiGLU constructed by the base
            transformer block.
        layer_initializer_range (float): Width-adjusted initialization scale.
        original_hidden_size (int): Model width restored after expert creation.
        original_initializer_range (float): Model initialization scale restored
            after expert creation.
        use_padding_free_transformer (bool): Native VWT packed-input flag.
        sequence_parallel (bool): Tensor-parallel sequence-sharding flag.

    Returns:
        nn.Module: Dense, legacy shared/private, or pooled-expert MLP.
    """

    mlp_config = config.mlp_blocks[layer_idx]
    if config.attention_group_moe:
        private_intermediate_size = (
            config.attention_group_moe_private_intermediate_sizes[layer_idx]
        )

        def make_private_expert() -> nn.Module:
            """Build one independently initialized private VWT expert."""

            config.hidden_size = hidden_size
            config.initializer_range = layer_initializer_range
            original_intermediate_size = mlp_config.intermediate_size
            mlp_config.intermediate_size = private_intermediate_size
            try:
                return get_mlp_block(
                    config,
                    use_padding_free_transformer,
                    sequence_parallel,
                    layer_idx,
                )
            finally:
                mlp_config.intermediate_size = original_intermediate_size
                config.hidden_size = original_hidden_size
                config.initializer_range = original_initializer_range

        return AttentionGroupMoE(
            hidden_size=hidden_size,
            num_private_experts=num_groups,
            shared_expert=dense_expert,
            private_expert_factory=make_private_expert,
            router_std=float(dense_expert.c_fc.std),
        )

    if not config.attention_group_pooled_moe or not uses_attention_groups:
        return dense_expert

    public_intermediate_size = (
        config.attention_group_pooled_moe_public_intermediate_sizes[layer_idx]
    )
    private_intermediate_size = (
        config.attention_group_pooled_moe_private_intermediate_sizes[layer_idx]
    )

    def make_pool(
        *,
        intermediate_size: int,
        num_experts: int,
        experts_per_token: int,
        auxiliary_loss_scale: float,
    ) -> MaskedExpertPool:
        """Build one independently initialized sparse expert pool.

        Args:
            intermediate_size (int): Per-expert SwiGLU width.
            num_experts (int): Stored experts.
            experts_per_token (int): Selected experts per valid token.
            auxiliary_loss_scale (float): Pool auxiliary-loss multiplier.

        Returns:
            MaskedExpertPool: Native ScatterMoE-compatible pool.
        """

        return MaskedExpertPool(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            shared_intermediate_size=None,
            use_interleaved_weights_for_shared_experts=False,
            use_interleaved_weights=mlp_config.use_interleaved_weights,
            shared_expert_gating=False,
            normalized_topk=True,
            num_experts=num_experts,
            num_experts_per_tok=experts_per_token,
            add_bias=mlp_config.add_bias,
            activation_function=mlp_config.activation_function,
            dropout=mlp_config.dropout,
            init_method=config.init_method,
            initializer_range=layer_initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            use_depth_scaled_init=config.use_depth_scaled_init,
            use_padding_free_transformer=False,
            sequence_parallel=sequence_parallel,
            auxiliary_loss_scale=auxiliary_loss_scale,
        )

    public_pool = make_pool(
        intermediate_size=public_intermediate_size,
        num_experts=config.attention_group_public_experts,
        experts_per_token=config.attention_group_public_experts_per_token,
        auxiliary_loss_scale=1.0,
    )

    def make_private_pool() -> MaskedExpertPool:
        """Build one independently initialized group-private pool."""

        return make_pool(
            intermediate_size=private_intermediate_size,
            num_experts=config.attention_group_private_experts_per_group,
            experts_per_token=config.attention_group_private_experts_per_token,
            auxiliary_loss_scale=1.0 / num_groups,
        )

    return AttentionGroupPooledMoE(
        hidden_size=hidden_size,
        num_private_groups=num_groups,
        public_pool=public_pool,
        private_pool_factory=make_private_pool,
        router_std=float(dense_expert.c_fc.std),
    )
