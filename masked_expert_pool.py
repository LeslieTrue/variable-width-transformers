"""Sparse native-VWT expert pools with valid-token capacity masking."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce

from lm_engine.hf_models.loss import add_aux_loss
from lm_engine.hf_models.modeling_utils.mlp_blocks.moe import MoE
from lm_engine.utils import ProcessGroupManager


class MaskedExpertPool(MoE):
    """Native VWT expert pool with optional valid-token compaction.

    This subclass preserves the released ScatterMoE parameterization and
    kernels while allowing attention-group capacity padding to be excluded
    from both expert selection and the auxiliary loss.

    Attributes:
        auxiliary_loss_scale (float): Multiplier applied to this pool's native
            Switch-plus-z auxiliary loss.
    """

    auxiliary_loss_scale: float

    def __init__(self, *, auxiliary_loss_scale: float = 1.0, **kwargs) -> None:
        """Initialize a masked native-VWT expert pool.

        Args:
            auxiliary_loss_scale (float): Nonnegative multiplier for this
                pool's routing auxiliary loss.
            **kwargs: Native :class:`MoE` constructor arguments.
        """

        if auxiliary_loss_scale < 0:
            raise ValueError("auxiliary_loss_scale must be nonnegative")
        super().__init__(**kwargs)
        self.auxiliary_loss_scale = auxiliary_loss_scale

    def _masked_switch_loss(
        self,
        *,
        logits: torch.Tensor,
        selected_experts: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute sequence-granular routing loss over valid compact tokens.

        Args:
            logits (torch.Tensor): Valid-token router logits, shape
                ``[valid_tokens, num_experts]``.
            selected_experts (torch.Tensor): Selected expert ids, shape
                ``[valid_tokens, top_k]``.
            batch_indices (torch.Tensor): Source sequence id for each valid
                token, shape ``[valid_tokens]``.
            batch_size (int): Physical microbatch size.

        Returns:
            torch.Tensor: Per-sequence Switch-plus-z loss, shape ``[batch]``.
        """

        probabilities = F.softmax(logits.float(), dim=-1).type_as(logits)
        accumulated_probabilities = logits.new_zeros(batch_size, self.num_experts)
        accumulated_probabilities.index_add_(0, batch_indices, probabilities)

        selected_counts = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).sum(dim=1)
        expert_frequency = logits.new_zeros(batch_size, self.num_experts)
        expert_frequency.index_add_(
            0, batch_indices, selected_counts.to(logits.dtype)
        )
        if (
            ProcessGroupManager.is_initialized()
            and ProcessGroupManager.get_data_parallel_world_size() > 1
        ):
            expert_frequency = all_reduce(
                expert_frequency,
                reduceOp="sum",
                group=ProcessGroupManager.get_data_parallel_group(),
            )

        switch_terms = F.normalize(
            accumulated_probabilities, p=1, dim=-1
        ) * F.normalize(expert_frequency.float(), p=1, dim=-1)
        switch_loss = self.num_experts * switch_terms.sum(dim=-1)

        z_values = torch.logsumexp(logits, dim=-1).square()
        z_sums = logits.new_zeros(batch_size)
        z_sums.index_add_(0, batch_indices, z_values)
        token_counts = logits.new_zeros(batch_size)
        token_counts.index_add_(
            0, batch_indices, torch.ones_like(batch_indices, dtype=logits.dtype)
        )
        z_loss = z_sums / token_counts.clamp_min(1)
        return (switch_loss + 0.1 * z_loss).type_as(logits)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route valid tokens through the sparse expert pool.

        Args:
            x (torch.Tensor): Token representations, shape
                ``[batch, sequence, hidden]``.
            valid_mask (torch.Tensor | None): Optional boolean mask with shape
                ``[batch, sequence]``. Invalid capacity-padding rows produce
                zero output and do not enter routing statistics.

        Returns:
            torch.Tensor: Routed updates with shape
            ``[batch, sequence, hidden]``.
        """

        if valid_mask is None:
            return super().forward(x)
        if x.ndim != 3 or valid_mask.shape != x.shape[:2]:
            raise ValueError("valid_mask must match the first two input dimensions")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        if self.is_tp_enabled:
            raise ValueError("masked expert pools do not support tensor parallelism")

        batch_size, sequence_length, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"expert pool expected width {self.hidden_size}, got {hidden_size}"
            )
        flat_x = x.reshape(batch_size * sequence_length, hidden_size)
        valid_indices = valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            if self.training and not self.loss_free_balancing:
                add_aux_loss(x.new_zeros(batch_size))
            return torch.zeros_like(x)

        valid_x = flat_x.index_select(0, valid_indices)
        logits, router_weights, selected_experts = self._compute_routing_weights(
            valid_x
        )
        valid_output, _ = self._compute_experts(
            valid_x, router_weights, selected_experts
        )
        valid_output = self.dropout(valid_output)
        batch_indices = torch.div(
            valid_indices, sequence_length, rounding_mode="floor"
        )
        if self.training and not self.loss_free_balancing:
            auxiliary_loss = self._masked_switch_loss(
                logits=logits,
                selected_experts=selected_experts,
                batch_indices=batch_indices,
                batch_size=batch_size,
            )
            add_aux_loss(auxiliary_loss * self.auxiliary_loss_scale)

        flat_output = torch.zeros_like(flat_x)
        flat_output.index_add_(0, valid_indices, valid_output)
        return flat_output.view(batch_size, sequence_length, hidden_size)
