"""Shared-router token-dispatch attention for variable-width transformers.

The grouped span routes every token once at its entrance, reuses that routing
decision across its layers, and restricts causal attention to tokens assigned
to the same group.  Group outputs are mixed with normalized top-k gates.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce

from lm_engine.hf_models.loss import add_aux_loss
from lm_engine.hf_models.modeling_utils.linear import ParameterizedLinear
from lm_engine.hf_models.modeling_utils.position_embedding.rope import (
    apply_rotary_pos_emb,
)
from lm_engine.hf_models.parameter import mark_parameter_as_mup_learning_rate
from lm_engine.utils import ProcessGroupManager


def dense_attention_flops(*, sequence_length: int, width: int) -> float:
    """Return VWT-convention forward FLOPs for one dense attention layer.

    Args:
        sequence_length (int): Number of tokens in each sequence.
        width (int): Residual width of the layer.

    Returns:
        float: QKV/output projection plus score/value-product FLOPs.
    """

    return 8.0 * sequence_length * width**2 + 4.0 * sequence_length**2 * width


def grouped_attention_flops(
    *,
    sequence_length: int,
    width: int,
    dense_heads: int,
    group_heads: int,
    num_groups: int,
    top_k: int,
) -> float:
    """Return ideal-uniform forward FLOPs for one grouped attention layer.

    Args:
        sequence_length (int): Number of tokens in each sequence.
        width (int): Residual width of the layer.
        dense_heads (int): Head count in the matching dense layer.
        group_heads (int): Head count owned by each attention group.
        num_groups (int): Number of attention groups.
        top_k (int): Groups selected per token.

    Returns:
        float: Ideal grouped projection and attention-core FLOPs.

    Assumptions:
        Every sequence assigns exactly ``top_k * sequence_length / num_groups``
        tokens to each group. Runtime capacity padding is intentionally omitted.
    """

    projection_ratio = top_k * group_heads / dense_heads
    core_ratio = top_k**2 * group_heads / (num_groups * dense_heads)
    return (
        8.0 * sequence_length * width**2 * projection_ratio
        + 4.0 * sequence_length**2 * width * core_ratio
    )


def shared_router_flops(
    *, sequence_length: int, input_width: int, num_groups: int
) -> float:
    """Return linear-router FLOPs for one shared routing decision.

    Args:
        sequence_length (int): Number of tokens in each sequence.
        input_width (int): Width of the grouped span entrance.
        num_groups (int): Number of routing logits per token.

    Returns:
        float: Forward FLOPs of the bias-free router projection.
    """

    return 2.0 * sequence_length * input_width * num_groups


def resolve_attention_group_head_schedule(
    *,
    widths: list[int],
    dense_heads: list[int],
    sequence_length: int,
    num_groups: int,
    top_k: int,
    compute_match: bool,
) -> list[int]:
    """Choose per-layer integer group heads, optionally matching dense FLOPs.

    Args:
        widths (list[int]): Grouped-layer widths, shape ``[num_grouped_layers]``.
        dense_heads (list[int]): Matching dense head counts per layer.
        sequence_length (int): Training context length.
        num_groups (int): Number of attention groups.
        top_k (int): Groups selected per token.
        compute_match (bool): Whether to reinvest ideal attention-core savings.

    Returns:
        list[int]: Heads owned by each group in every grouped layer.

    Raises:
        ValueError: If shapes or routing geometry are invalid.

    Notes:
        The compute-matched solution enumerates the floor/ceiling choices around
        every layer's continuous optimum. At the released depths this is at most
        ``2**11`` schedules, so exhaustive selection is deterministic and cheap.
    """

    if not widths or len(widths) != len(dense_heads):
        raise ValueError(
            "widths and dense_heads must be nonempty and have equal length"
        )
    if sequence_length <= 0 or num_groups <= 1 or not 1 <= top_k <= num_groups:
        raise ValueError("invalid attention-group geometry")
    if any(width <= 0 for width in widths) or any(heads <= 0 for heads in dense_heads):
        raise ValueError("widths and head counts must be positive")
    if any(width % heads != 0 for width, heads in zip(widths, dense_heads)):
        raise ValueError(
            "each grouped-layer width must be divisible by its dense head count"
        )

    if not compute_match:
        return [max(1, round(heads / top_k)) for heads in dense_heads]

    candidates: list[tuple[int, ...]] = []
    for width, heads in zip(widths, dense_heads):
        numerator = heads * (2.0 * width + sequence_length)
        denominator = 2.0 * top_k * width + sequence_length * top_k**2 / num_groups
        ideal = numerator / denominator
        lower = max(1, math.floor(ideal))
        upper = max(1, math.ceil(ideal))
        candidates.append(tuple(sorted({lower, upper})))

    target = sum(
        dense_attention_flops(sequence_length=sequence_length, width=width)
        for width in widths
    )
    router = shared_router_flops(
        sequence_length=sequence_length,
        input_width=widths[0],
        num_groups=num_groups,
    )

    def schedule_key(schedule: tuple[int, ...]) -> tuple[float, int, tuple[int, ...]]:
        grouped = router + sum(
            grouped_attention_flops(
                sequence_length=sequence_length,
                width=width,
                dense_heads=heads,
                group_heads=group_heads,
                num_groups=num_groups,
                top_k=top_k,
            )
            for width, heads, group_heads in zip(widths, dense_heads, schedule)
        )
        parameter_proxy = sum(
            num_groups * group_heads * width**2 // heads
            for width, heads, group_heads in zip(widths, dense_heads, schedule)
        )
        return abs(grouped - target), parameter_proxy, schedule

    return list(min(itertools.product(*candidates), key=schedule_key))


def attention_group_flop_ratio(
    *,
    widths: list[int],
    dense_heads: list[int],
    group_heads: list[int],
    sequence_length: int,
    num_groups: int,
    top_k: int,
) -> float:
    """Return grouped-span FLOPs divided by its dense counterpart.

    Args:
        widths (list[int]): Grouped-layer widths.
        dense_heads (list[int]): Dense heads per grouped layer.
        group_heads (list[int]): Derived heads per group and layer.
        sequence_length (int): Context length.
        num_groups (int): Number of attention groups.
        top_k (int): Groups selected per token.

    Returns:
        float: Ideal grouped-to-dense FLOP ratio including the shared router.
    """

    if not (len(widths) == len(dense_heads) == len(group_heads)):
        raise ValueError("all per-layer lists must have equal length")
    dense = sum(
        dense_attention_flops(sequence_length=sequence_length, width=width)
        for width in widths
    )
    grouped = shared_router_flops(
        sequence_length=sequence_length,
        input_width=widths[0],
        num_groups=num_groups,
    ) + sum(
        grouped_attention_flops(
            sequence_length=sequence_length,
            width=width,
            dense_heads=heads,
            group_heads=group_head_count,
            num_groups=num_groups,
            top_k=top_k,
        )
        for width, heads, group_head_count in zip(widths, dense_heads, group_heads)
    )
    return grouped / dense


@dataclass
class AttentionGroupDispatchEntry:
    """Compact dispatch tensors for one attention group.

    Attributes:
        gather_indices (torch.Tensor): Source positions, shape ``[batch, capacity]``.
        valid_mask (torch.Tensor): Real compact rows, shape ``[batch, capacity]``.
        gates (torch.Tensor): Mixture weights, shape ``[batch, capacity]``.
        attention_mask (torch.Tensor): Causal compact mask, shape
            ``[batch, 1, capacity, capacity]``.
    """

    gather_indices: torch.Tensor
    valid_mask: torch.Tensor
    gates: torch.Tensor
    attention_mask: torch.Tensor


@dataclass
class AttentionGroupDispatchPlan:
    """Routing decision and compact layouts reused across grouped layers.

    Attributes:
        entries (list[AttentionGroupDispatchEntry]): Per-group compact layouts.
        assignment_shares (torch.Tensor): Normalized assignment counts, shape
            ``[num_groups]``.
        entropy (torch.Tensor): Entropy of normalized group assignment shares.
        balance (torch.Tensor): Minimum share divided by maximum share.
        max_capacity_ratio (torch.Tensor): Largest compact capacity divided by
            the ideal uniform membership count.
    """

    entries: list[AttentionGroupDispatchEntry]
    assignment_shares: torch.Tensor
    entropy: torch.Tensor
    balance: torch.Tensor
    max_capacity_ratio: torch.Tensor


class AttentionGroupRouter(nn.Module):
    """Bias-free shared top-k router with VWT switch/z auxiliary loss.

    Attributes:
        input_width (int): Router input width.
        num_groups (int): Number of attention groups.
        top_k (int): Groups selected per token.
        capacity_multiple (int): Compact capacity alignment.
        gate (ParameterizedLinear): Bias-free routing projection.
    """

    input_width: int
    num_groups: int
    top_k: int
    capacity_multiple: int

    def __init__(
        self,
        *,
        input_width: int,
        num_groups: int,
        top_k: int,
        capacity_multiple: int,
        std: float,
    ) -> None:
        """Initialize the shared group router.

        Args:
            input_width (int): Width at the grouped span entrance.
            num_groups (int): Number of routing choices.
            top_k (int): Choices selected per token.
            capacity_multiple (int): Alignment for compact sequence capacity.
            std (float): VWT-compatible router weight initialization standard deviation.
        """

        super().__init__()
        if input_width <= 0 or num_groups <= 1 or not 1 <= top_k <= num_groups:
            raise ValueError("invalid shared attention-group router geometry")
        if capacity_multiple <= 0:
            raise ValueError("capacity_multiple must be positive")
        self.input_width = input_width
        self.num_groups = num_groups
        self.top_k = top_k
        self.capacity_multiple = capacity_multiple
        self.gate = ParameterizedLinear(input_width, num_groups, bias=False, std=std)
        mark_parameter_as_mup_learning_rate(self.gate.weight)

    def _compute_auxiliary_loss(
        self,
        *,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        selected_groups: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the sequence-granular VWT switch plus z-loss objective.

        Args:
            logits (torch.Tensor): Router logits, shape ``[batch, seq, groups]``.
            probabilities (torch.Tensor): Full softmax, same shape as ``logits``.
            selected_groups (torch.Tensor): Top-k ids, shape ``[batch, seq, top_k]``.

        Returns:
            torch.Tensor: Per-sequence auxiliary loss, shape ``[batch]``.
        """

        frequency = (
            F.one_hot(selected_groups, num_classes=self.num_groups)
            .sum(dim=(1, 2))
            .float()
        )
        if (
            ProcessGroupManager.is_initialized()
            and ProcessGroupManager.get_data_parallel_world_size() > 1
        ):
            frequency = all_reduce(
                frequency,
                reduceOp="sum",
                group=ProcessGroupManager.get_data_parallel_group(),
            )
        accumulated_probabilities = probabilities.sum(dim=1)
        switch_terms = F.normalize(
            accumulated_probabilities, p=1, dim=-1
        ) * F.normalize(frequency, p=1, dim=-1)
        switch_loss = self.num_groups * switch_terms.sum(dim=-1)
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean(dim=1)
        return (switch_loss + 0.1 * z_loss).type_as(logits)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> AttentionGroupDispatchPlan:
        """Route tokens and build a no-drop compact causal dispatch plan.

        Args:
            x (torch.Tensor): Grouped-span input, shape ``[batch, seq, input_width]``.

        Returns:
            AttentionGroupDispatchPlan: Routing tensors shared by grouped layers.

        Assumptions:
            Inputs are fixed-length VWT pretraining sequences without padding.
        """

        batch_size, sequence_length, width = x.shape
        if width != self.input_width:
            raise ValueError(f"router expected width {self.input_width}, got {width}")

        logits = self.gate(x)
        probabilities = F.softmax(logits.float(), dim=-1).type_as(x)
        topk_logits, selected_groups = logits.topk(self.top_k, dim=-1)
        topk_gates = F.softmax(topk_logits.float(), dim=-1).type_as(x)
        gates = logits.new_zeros(batch_size, sequence_length, self.num_groups)
        gates.scatter_add_(dim=-1, index=selected_groups, src=topk_gates)
        memberships = gates > 0
        counts_by_batch_group = memberships.sum(dim=1)  # [batch, groups]
        max_counts = counts_by_batch_group.amax(dim=0).tolist()

        if self.training:
            add_aux_loss(
                self._compute_auxiliary_loss(
                    logits=logits,
                    probabilities=probabilities,
                    selected_groups=selected_groups,
                )
            )

        entries: list[AttentionGroupDispatchEntry] = []
        capacities: list[int] = []
        for group_index in range(self.num_groups):
            members = memberships[..., group_index]  # [batch, seq]
            counts = counts_by_batch_group[:, group_index]  # [batch]
            max_count = int(max_counts[group_index])
            if max_count == 0:
                capacity = 0
                gather_indices = torch.empty(
                    batch_size, 0, dtype=torch.long, device=x.device
                )
                valid = torch.empty(batch_size, 0, dtype=torch.bool, device=x.device)
                compact_gates = x.new_empty(batch_size, 0)
                compact_mask = torch.empty(
                    batch_size, 1, 0, 0, dtype=torch.bool, device=x.device
                )
            else:
                capacity = min(
                    sequence_length,
                    math.ceil(max_count / self.capacity_multiple)
                    * self.capacity_multiple,
                )
                order = torch.argsort(
                    members.to(torch.int8), dim=1, descending=True, stable=True
                )
                gather_indices = order[:, :capacity]
                valid = (
                    torch.arange(capacity, device=x.device)[None, :] < counts[:, None]
                )
                compact_gates = torch.gather(
                    gates[..., group_index], dim=1, index=gather_indices
                )
                causal = torch.ones(
                    capacity, capacity, dtype=torch.bool, device=x.device
                ).tril()
                valid_keys = (
                    causal[None, None]
                    & valid[:, None, :, None]
                    & valid[:, None, None, :]
                )
                padded_query_diagonal = (
                    ~valid[:, None, :, None]
                    & torch.eye(capacity, dtype=torch.bool, device=x.device)[None, None]
                )
                compact_mask = valid_keys | padded_query_diagonal
            capacities.append(capacity)
            entries.append(
                AttentionGroupDispatchEntry(
                    gather_indices=gather_indices,
                    valid_mask=valid,
                    gates=compact_gates,
                    attention_mask=compact_mask,
                )
            )

        assignment_counts = memberships.sum(dim=(0, 1)).float()
        assignment_shares = assignment_counts / assignment_counts.sum().clamp_min(1.0)
        entropy = -(assignment_shares * assignment_shares.clamp_min(1e-12).log()).sum()
        balance = assignment_shares.min() / assignment_shares.max().clamp_min(1e-12)
        ideal_capacity = self.top_k * sequence_length / self.num_groups
        max_capacity_ratio = x.new_tensor(max(capacities, default=0) / ideal_capacity)
        return AttentionGroupDispatchPlan(
            entries=entries,
            assignment_shares=assignment_shares.detach(),
            entropy=entropy.detach(),
            balance=balance.detach(),
            max_capacity_ratio=max_capacity_ratio.detach(),
        )


class _AttentionGroupProjection(nn.Module):
    """QKV and output projections owned by one attention group.

    Attributes:
        hidden_size (int): Input and output width.
        num_heads (int): Heads owned by this group.
        head_dim (int): Width of every head.
        qkv (ParameterizedLinear): Combined query/key/value projection.
        out_proj (ParameterizedLinear): Group output projection.
    """

    hidden_size: int
    num_heads: int
    head_dim: int

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        add_bias: bool,
        qkv_std: float,
        out_std: float,
    ) -> None:
        """Initialize one group's projections.

        Args:
            hidden_size (int): Residual width.
            num_heads (int): Group head count.
            head_dim (int): Width of each head.
            add_bias (bool): Whether projections include biases.
            qkv_std (float): Input projection initialization standard deviation.
            out_std (float): Output projection initialization standard deviation.
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.qkv = ParameterizedLinear(
            hidden_size, 3 * inner_dim, bias=add_bias, std=qkv_std
        )
        self.out_proj = ParameterizedLinear(
            inner_dim, hidden_size, bias=add_bias, std=out_std
        )
        mark_parameter_as_mup_learning_rate(self.qkv.weight)
        mark_parameter_as_mup_learning_rate(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        scale: float,
        dropout_p: float,
    ) -> torch.Tensor:
        """Run one compact causal group attention operation.

        Args:
            x (torch.Tensor): Compact inputs, shape ``[batch, capacity, hidden]``.
            rope_cos_sin (tuple[torch.Tensor, torch.Tensor]): Compact RoPE tensors,
                each shape ``[batch, capacity, head_dim]``.
            attention_mask (torch.Tensor): Boolean mask, shape
                ``[batch, 1, capacity, capacity]``.
            scale (float): Softmax scale.
            dropout_p (float): Training-time attention dropout probability.

        Returns:
            torch.Tensor: Compact group output, shape ``[batch, capacity, hidden]``.
        """

        batch_size, capacity, _ = x.shape
        inner_dim = self.num_heads * self.head_dim
        q, k, v = self.qkv(x).split(inner_dim, dim=-1)
        shape = (batch_size, capacity, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q.view(shape), cos_sin=rope_cos_sin).transpose(1, 2)
        k = apply_rotary_pos_emb(k.view(shape), cos_sin=rope_cos_sin).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )
        output = output.transpose(1, 2).reshape(batch_size, capacity, inner_dim)
        return self.out_proj(output)


class GroupedSelfAttention(nn.Module):
    """Per-group compact causal attention mixed by a shared routing plan.

    Attributes:
        hidden_size (int): Residual width.
        num_groups (int): Number of independently parameterized groups.
        group_heads (int): Heads per group at this layer.
        head_dim (int): Dense-compatible head dimension.
        groups (nn.ModuleList): Per-group QKV/output projections.
    """

    hidden_size: int
    num_groups: int
    group_heads: int
    head_dim: int

    def __init__(
        self,
        *,
        hidden_size: int,
        num_groups: int,
        group_heads: int,
        head_dim: int,
        attention_multiplier: float,
        add_bias: bool,
        softmax_dropout: float,
        dropout: float,
        qkv_std: float,
        out_std: float,
    ) -> None:
        """Initialize all attention groups at one transformer layer.

        Args:
            hidden_size (int): Residual width.
            num_groups (int): Number of attention parameter banks.
            group_heads (int): Heads in each bank.
            head_dim (int): Width of each head.
            attention_multiplier (float): Softmax scale.
            add_bias (bool): Whether projection layers use bias.
            softmax_dropout (float): Attention-probability dropout.
            dropout (float): Output dropout.
            qkv_std (float): QKV initialization standard deviation.
            out_std (float): Output initialization standard deviation.
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_heads = group_heads
        self.head_dim = head_dim
        self.attention_multiplier = attention_multiplier
        self.softmax_dropout = softmax_dropout
        self.dropout = dropout
        self.groups = nn.ModuleList(
            [
                _AttentionGroupProjection(
                    hidden_size=hidden_size,
                    num_heads=group_heads,
                    head_dim=head_dim,
                    add_bias=add_bias,
                    qkv_std=qkv_std,
                    out_std=out_std,
                )
                for _ in range(num_groups)
            ]
        )

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor],
        dispatch_plan: AttentionGroupDispatchPlan,
    ) -> torch.Tensor:
        """Apply all active groups without dropping routed tokens.

        Args:
            x (torch.Tensor): Normalized layer input, shape ``[batch, seq, hidden]``.
            rope_cos_sin (tuple[torch.Tensor, torch.Tensor]): Full-sequence RoPE
                tensors, each shape ``[batch, seq, head_dim]``.
            dispatch_plan (AttentionGroupDispatchPlan): Shared compact routing plan.

        Returns:
            torch.Tensor: Gate-weighted output, shape ``[batch, seq, hidden]``.
        """

        if len(dispatch_plan.entries) != self.num_groups:
            raise ValueError("dispatch plan group count does not match the layer")
        batch_size, sequence_length, _ = x.shape
        output = x.new_zeros(batch_size, sequence_length, self.hidden_size)
        cos, sin = rope_cos_sin
        if cos.shape[0] == 1 and batch_size > 1:
            cos = cos.expand(batch_size, -1, -1)
            sin = sin.expand(batch_size, -1, -1)
        for group, entry in zip(self.groups, dispatch_plan.entries):
            capacity = entry.gather_indices.shape[1]
            if capacity == 0:
                probe_qkv = group.qkv(x[:, :1])
                inner_dim = group.num_heads * group.head_dim
                probe = group.out_proj(probe_qkv[..., :inner_dim])
                output = output + 0.0 * probe.sum()
                continue
            gather_index = entry.gather_indices.unsqueeze(-1).expand(
                -1, -1, self.hidden_size
            )
            compact_x = torch.gather(x, dim=1, index=gather_index)
            compact_cos = torch.gather(
                cos,
                dim=1,
                index=entry.gather_indices.unsqueeze(-1).expand(-1, -1, cos.shape[-1]),
            )
            compact_sin = torch.gather(
                sin,
                dim=1,
                index=entry.gather_indices.unsqueeze(-1).expand(-1, -1, sin.shape[-1]),
            )
            compact_output = group(
                compact_x,
                rope_cos_sin=(compact_cos, compact_sin),
                attention_mask=entry.attention_mask,
                scale=self.attention_multiplier,
                dropout_p=self.softmax_dropout if self.training else 0.0,
            )
            compact_output = torch.where(
                entry.valid_mask.unsqueeze(-1), compact_output, 0.0
            )
            compact_output = compact_output * entry.gates.unsqueeze(-1)
            output.scatter_add_(dim=1, index=gather_index, src=compact_output)
        return F.dropout(output, p=self.dropout, training=self.training)
