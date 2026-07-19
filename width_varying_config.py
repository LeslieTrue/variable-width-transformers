import logging
import random

from lm_engine.hf_models.config import CommonConfig
from lm_engine.utils import log_rank_0

from attention_grouping import resolve_attention_group_head_schedule

from scripts.solve_hparams import (
    compute_base_width,
    solve_geometric_factors_for_bottleneck_ratio,
    SolverConfig,
    ModelParams,
)


def _compute_missing_factor_geometric(
    expansion_factor: float | None,
    reduction_factor: float | None,
    symmetric_widths: bool,
    max_layer: int,
    num_layers: int,
    duplicate_max_layer: bool,
) -> tuple[float, float]:
    """
    Compute the missing expansion or reduction factor for geometric schedule.

    Args:
        expansion_factor: Provided expansion factor, or None if to be computed.
        reduction_factor: Provided reduction factor, or None if to be computed.
        symmetric_widths: If True, compute so w_1 = w_L. If False, compute as 1/provided_factor.
        max_layer: Layer index of the peak width.
        num_layers: Total number of layers.
        duplicate_max_layer: Whether the peak layer is duplicated.

    Returns:
        Tuple of (expansion_factor, reduction_factor).
    """
    if expansion_factor is not None and reduction_factor is not None:
        return expansion_factor, reduction_factor

    grow_steps = max_layer - 1
    shrink_steps = num_layers - max_layer - (1 if duplicate_max_layer else 0)
    assert shrink_steps >= 0

    if expansion_factor is not None:
        # Compute reduction_factor from expansion_factor
        if symmetric_widths and shrink_steps > 0:
            # For w_1 = w_L: expansion^grow_steps * reduction^shrink_steps = 1
            reduction_factor = expansion_factor ** (-grow_steps / shrink_steps)
        else:
            reduction_factor = 1.0 / expansion_factor
    else:
        # Compute expansion_factor from reduction_factor
        if symmetric_widths and grow_steps > 0:
            expansion_factor = reduction_factor ** (-shrink_steps / grow_steps)
        else:
            expansion_factor = 1.0 / reduction_factor

    return expansion_factor, reduction_factor


def _compute_missing_delta_arithmetic(
    delta: float | None,
    delta_neg: float | None,
    symmetric_widths: bool,
    max_layer: int,
    num_layers: int,
    duplicate_max_layer: bool,
) -> tuple[float, float]:
    """
    Compute the missing delta or delta_neg for arithmetic schedule.

    Args:
        delta: Provided growth delta, or None if to be computed.
        delta_neg: Provided shrink delta, or None if to be computed.
        symmetric_widths: If True, compute so w_1 = w_L. If False, compute as -provided_delta.
        max_layer: Layer index of the peak width.
        num_layers: Total number of layers.
        duplicate_max_layer: Whether the peak layer is duplicated.

    Returns:
        Tuple of (delta, delta_neg).
    """
    if delta is not None and delta_neg is not None:
        return delta, delta_neg

    grow_steps = max_layer - 1
    shrink_steps = num_layers - max_layer - (1 if duplicate_max_layer else 0)
    assert shrink_steps >= 0

    if delta is not None:
        # Compute delta_neg from delta
        if symmetric_widths and shrink_steps > 0:
            # For w_1 = w_L: delta * grow_steps + delta_neg * shrink_steps = 0
            delta_neg = -delta * grow_steps / shrink_steps
        else:
            delta_neg = -delta
    else:
        # Compute delta from delta_neg
        if symmetric_widths and grow_steps > 0:
            delta = -delta_neg * shrink_steps / grow_steps
        else:
            delta = -delta_neg

    return delta, delta_neg


class WidthVaryingConfig(CommonConfig):
    model_type = "width_varying"

    def __init__(
        self,
        base_width: int | None = None,
        bottleneck_ratio: float | None = None,
        expansion_factor: float | None = None,
        reduction_factor: float | None = None,
        max_layer: int = 1,
        duplicate_max_layer: bool = False,
        quantize_to: int = 1,
        wide_unembedding: bool = False,
        expand_method: str = "zero",
        interpolate_align_corners: bool = False,
        fixed_residual_width: bool = False,
        original_input_width: bool = True,
        original_output_width: bool = True,
        permute_widths_seed: int | str | None = None,
        permute_all_widths: bool = False,
        symmetric_widths: bool = False,
        delta: float | None = None,
        delta_neg: float | None = None,
        sinkhorn_iters: int = 0,
        expand_linear_bias: bool = False,
        variable_initialization: bool = True,
        attention_num_groups: int = 1,
        attention_num_groups_per_token: int = 1,
        attention_group_num_layers: int = 0,
        attention_group_compute_match: bool = False,
        attention_group_capacity_multiple: int = 128,
        attention_group_heads_per_layer: list[int] | None = None,
        attention_group_moe: bool = False,
        attention_group_moe_expansion_ratio: float = 2.0,
        **kwargs,
    ) -> None:
        # Automatically determine schedule type based on which parameters are provided
        has_geometric_factors = (expansion_factor is not None) or (reduction_factor is not None)
        has_bottleneck_ratio = bottleneck_ratio is not None
        has_geometric = has_geometric_factors or has_bottleneck_ratio
        has_arithmetic = (delta is not None) or (delta_neg is not None)
        dummy_run = (
            base_width is None
            and max_layer == 1
            and quantize_to == 1
            and not has_geometric
            and not has_arithmetic
        )  # this is a special logging mode, see self.__class__() in transformers/configuration_utils.py

        if not dummy_run:
            assert has_geometric or has_arithmetic, \
                "Must provide either expansion_factor/reduction_factor (geometric) or delta/delta_neg (arithmetic)"
            assert not (has_geometric and has_arithmetic), \
                "Cannot provide both geometric (expansion_factor/reduction_factor) and arithmetic (delta/delta_neg) parameters"

        self.base_width = base_width
        self.bottleneck_ratio = bottleneck_ratio
        self.max_layer = max_layer
        self.duplicate_max_layer = duplicate_max_layer
        self.quantize_to = quantize_to
        self.wide_unembedding = wide_unembedding
        self.permute_widths_seed = permute_widths_seed
        self.permute_all_widths = permute_all_widths
        self.symmetric_widths = symmetric_widths
        self.schedule_type = "arithmetic" if has_arithmetic else "geometric"

        if permute_all_widths:
            assert original_input_width and original_output_width, (
                "permute_all_widths requires original_input_width and original_output_width"
            )

        assert expand_method in [
            "zero",
            "gaussian",
            "move_up",
            "move_up_gated",
            "transposed_conv",
            "interpolate",
            "linear",
            "linear_diff",
            "conv_diff",
        ]
        self.expand_method = expand_method
        self.interpolate_align_corners = interpolate_align_corners
        self.sinkhorn_iters = sinkhorn_iters
        self.expand_linear_bias = expand_linear_bias
        self.variable_initialization = variable_initialization
        self.attention_num_groups = attention_num_groups
        self.attention_num_groups_per_token = attention_num_groups_per_token
        self.attention_group_num_layers = attention_group_num_layers
        self.attention_group_compute_match = attention_group_compute_match
        self.attention_group_capacity_multiple = attention_group_capacity_multiple
        self.attention_group_heads_per_layer = attention_group_heads_per_layer
        self.attention_group_moe = attention_group_moe
        self.attention_group_moe_expansion_ratio = attention_group_moe_expansion_ratio
        if sinkhorn_iters > 0:
            assert expand_method in ["linear", "linear_diff"], (
                f"sinkhorn_iters > 0 only supported with linear/linear_diff, got {expand_method}"
            )
            assert not expand_linear_bias, (
                "sinkhorn_iters > 0 requires expand_linear_bias=False"
            )
        self.fixed_residual_width = fixed_residual_width
        self.original_input_width = original_input_width
        self.original_output_width = original_output_width
        stored_rope_dim = kwargs.pop("rope_dim", None)
        assert stored_rope_dim in (None, -1), (
            "WidthVaryingConfig computes rope_dim per layer; do not pass a global rope_dim"
        )
        kwargs["rope_dim"] = -1  # to catch errors; if None, the parent class logic will break

        super().__init__(**kwargs)

        if dummy_run:
            return

        first_mlp = self.mlp_blocks[0] if self.mlp_blocks else None
        use_moe_for_solver = (
            first_mlp is not None
            and getattr(first_mlp, "mlp_type", None) == "MoE"
        )
        model_params = ModelParams(
            vocab_size=self.vocab_size,
            sequence_length=self.max_position_embeddings,
            mlp_expansion=4,  # Standard MLP expansion factor (ignored when MoE)
            num_mlp_projections=3,  # SwiGLU
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_experts=getattr(first_mlp, "num_experts", None) if use_moe_for_solver else None,
            num_experts_per_tok=getattr(first_mlp, "num_experts_per_tok", None) if use_moe_for_solver else None,
            moe_intermediate_size=getattr(self, "moe_intermediate_size", None) if use_moe_for_solver else None,
        )

        # Compute missing factors based on schedule type (needs self.num_layers from super().__init__())
        if self.schedule_type == "geometric":
            if bottleneck_ratio is not None and self.base_width is None:
                solver_config_for_ratio = SolverConfig(
                    schedule_type="geometric",
                    alpha=0.0,
                    alpha_neg=0.0,
                    k=self.max_layer,
                    duplicate_peak=self.duplicate_max_layer,
                    wide_unembedding=self.wide_unembedding,
                    constant_residual_stream=self.fixed_residual_width,
                    original_input_width=self.original_input_width,
                    original_output_width=self.original_output_width,
                )
                self.expansion_factor, self.reduction_factor = solve_geometric_factors_for_bottleneck_ratio(
                    model_params=model_params,
                    config=solver_config_for_ratio,
                    bottleneck_ratio=bottleneck_ratio,
                    symmetric_widths=symmetric_widths,
                )
                log_rank_0(
                    logging.INFO,
                    f"Computed factors from bottleneck_ratio={bottleneck_ratio:.6f}: "
                    f"expansion_factor={self.expansion_factor:.6f}, "
                    f"reduction_factor={self.reduction_factor:.6f}",
                )
            elif bottleneck_ratio is not None and has_geometric_factors:
                # Saved configs may include the computed base_width and factors.
                # In that case, reuse the stored factors instead of re-solving.
                self.expansion_factor, self.reduction_factor = _compute_missing_factor_geometric(
                    expansion_factor=expansion_factor,
                    reduction_factor=reduction_factor,
                    symmetric_widths=symmetric_widths,
                    max_layer=max_layer,
                    num_layers=self.num_layers,
                    duplicate_max_layer=duplicate_max_layer,
                )
            elif bottleneck_ratio is not None:
                raise ValueError(
                    "bottleneck_ratio with explicit base_width also requires expansion_factor/reduction_factor"
                )
            else:
                self.expansion_factor, self.reduction_factor = _compute_missing_factor_geometric(
                    expansion_factor=expansion_factor,
                    reduction_factor=reduction_factor,
                    symmetric_widths=symmetric_widths,
                    max_layer=max_layer,
                    num_layers=self.num_layers,
                    duplicate_max_layer=duplicate_max_layer,
                )
            self.delta = None
            self.delta_neg = None
            if bottleneck_ratio is None and (expansion_factor is None or reduction_factor is None):
                log_rank_0(
                    logging.INFO,
                    f"Computed missing factor: expansion_factor={self.expansion_factor:.6f}, "
                    f"reduction_factor={self.reduction_factor:.6f} (symmetric={symmetric_widths})",
                )
        else:  # arithmetic
            self.delta, self.delta_neg = _compute_missing_delta_arithmetic(
                delta=delta,
                delta_neg=delta_neg,
                symmetric_widths=symmetric_widths,
                max_layer=max_layer,
                num_layers=self.num_layers,
                duplicate_max_layer=duplicate_max_layer,
            )
            self.expansion_factor = None
            self.reduction_factor = None
            if delta is None or delta_neg is None:
                log_rank_0(
                    logging.INFO,
                    f"Computed missing delta: delta={self.delta:.6f}, "
                    f"delta_neg={self.delta_neg:.6f} (symmetric={symmetric_widths})",
                )

        # Compute base_width automatically if not provided
        if self.base_width is None:
            # Create config dataclass for solver based on schedule type
            if self.schedule_type == "geometric":
                # Compute alpha and alpha_neg from expansion_factor and reduction_factor
                # expansion_factor = 1 + alpha, so alpha = expansion_factor - 1
                alpha = self.expansion_factor - 1.0
                # reduction_factor = 1 + alpha_neg, so alpha_neg = reduction_factor - 1
                alpha_neg = self.reduction_factor - 1.0

                solver_config = SolverConfig(
                    schedule_type="geometric",
                    alpha=alpha,
                    k=self.max_layer,
                    alpha_neg=alpha_neg,
                    duplicate_peak=self.duplicate_max_layer,
                    wide_unembedding=self.wide_unembedding,
                    constant_residual_stream=self.fixed_residual_width,
                    original_input_width=self.original_input_width,
                    original_output_width=self.original_output_width,
                )
            else:  # arithmetic
                solver_config = SolverConfig(
                    schedule_type="arithmetic",
                    delta=self.delta,
                    k=self.max_layer,
                    delta_neg=self.delta_neg,
                    duplicate_peak=self.duplicate_max_layer,
                    wide_unembedding=self.wide_unembedding,
                    constant_residual_stream=self.fixed_residual_width,
                    original_input_width=self.original_input_width,
                    original_output_width=self.original_output_width,
                )

            # Automatically compute base_width from configuration
            computed_base_width = compute_base_width(
                model_params=model_params,
                config=solver_config,
                legacy_move_up_waste=bottleneck_ratio is None,
            )
            self.base_width = int(round(computed_base_width))
            log_rank_0(
                logging.INFO,
                f"Computed base_width automatically: {self.base_width:.2f} (from solver)",
            )

        if self.fixed_residual_width:
            assert (
                (not self.wide_unembedding)
                and self.original_input_width
                and self.original_output_width
            ), "wide_unembedding, original_input_width, and original_output_width must be disabled when fixed_residual_width is enabled"

        # Compute layer widths based on schedule type
        self.widths = []
        for layer_idx in range(1, self.num_layers + 1):
            if layer_idx == 1:
                width = self.base_width
            elif layer_idx <= self.max_layer:
                # Grow phase
                if self.schedule_type == "geometric":
                    width = self.widths[-1] * self.expansion_factor
                else:  # arithmetic
                    width = self.widths[-1] + self.base_width * self.delta
            elif self.duplicate_max_layer and layer_idx == self.max_layer + 1:
                # Duplicate peak layer
                width = self.widths[-1]
            else:
                # Shrink phase
                if self.schedule_type == "geometric":
                    width = self.widths[-1] * self.reduction_factor
                else:  # arithmetic
                    width = self.widths[-1] + self.base_width * self.delta_neg
            assert width > 0, f"Width must be positive, got {width} at layer {layer_idx}"
            self.widths.append(width)
        self.widths = [int(round(w / self.quantize_to) * self.quantize_to) for w in self.widths]

        # Permute or sort widths if specified
        if self.permute_widths_seed is not None:
            if self.permute_all_widths:
                target_widths = self.widths
                scope = "all layers"
            else:
                target_widths = self.widths[1:-1]
                scope = "middle layers only"

            if self.permute_widths_seed == "sort":
                target_widths = sorted(target_widths)
                log_rank_0(logging.INFO, f"Sorted widths ({scope}) in ascending order")
            elif self.permute_widths_seed == "rsort":
                target_widths = sorted(target_widths, reverse=True)
                log_rank_0(logging.INFO, f"Sorted widths ({scope}) in descending order")
            else:
                rng = random.Random(self.permute_widths_seed)
                rng.shuffle(target_widths)
                log_rank_0(
                    logging.INFO,
                    f"Permuted widths ({scope}) with seed {self.permute_widths_seed}",
                )

            if self.permute_all_widths:
                self.widths = target_widths
            else:
                self.widths = [self.widths[0]] + target_widths + [self.widths[-1]]

        grouping_enabled = self.attention_num_groups > 1
        if grouping_enabled:
            if not 1 <= self.attention_num_groups_per_token <= self.attention_num_groups:
                raise ValueError(
                    "attention_num_groups_per_token must be in [1, attention_num_groups]"
                )
            if not 1 <= self.attention_group_num_layers <= self.num_layers:
                raise ValueError(
                    "attention_group_num_layers must be in [1, num_layers] when grouping is enabled"
                )
            if self.fixed_residual_width:
                raise ValueError("attention grouping does not support fixed_residual_width")
            if self.attention_group_capacity_multiple <= 0:
                raise ValueError("attention_group_capacity_multiple must be positive")

            dense_heads = [
                block.num_attention_heads
                for block in self.sequence_mixer_blocks[: self.attention_group_num_layers]
            ]
            if self.attention_group_heads_per_layer is None:
                self.attention_group_heads_per_layer = resolve_attention_group_head_schedule(
                    widths=self.widths[: self.attention_group_num_layers],
                    dense_heads=dense_heads,
                    sequence_length=self.max_position_embeddings,
                    num_groups=self.attention_num_groups,
                    top_k=self.attention_num_groups_per_token,
                    compute_match=self.attention_group_compute_match,
                )
            elif len(self.attention_group_heads_per_layer) != self.attention_group_num_layers:
                raise ValueError(
                    "attention_group_heads_per_layer must have attention_group_num_layers entries"
                )
            if any(heads <= 0 for heads in self.attention_group_heads_per_layer):
                raise ValueError("attention group head counts must be positive")
            log_rank_0(
                logging.INFO,
                "Attention grouping: "
                f"layers={self.attention_group_num_layers}, "
                f"groups={self.attention_num_groups}, "
                f"top_k={self.attention_num_groups_per_token}, "
                f"heads={self.attention_group_heads_per_layer}",
            )
        else:
            if self.attention_group_num_layers != 0:
                raise ValueError(
                    "attention_group_num_layers must be 0 when attention_num_groups == 1"
                )
            self.attention_group_heads_per_layer = []

        if self.attention_group_moe:
            if not grouping_enabled:
                raise ValueError("attention_group_moe requires attention grouping")
            if self.num_layers < 3:
                raise ValueError("the three-block attention-group MoE needs at least 3 layers")
            if self.attention_group_moe_expansion_ratio <= 0:
                raise ValueError("attention_group_moe_expansion_ratio must be positive")
            for mlp_block in self.mlp_blocks:
                if getattr(mlp_block, "mlp_type", None) != "MLP":
                    raise ValueError("attention_group_moe requires native dense MLP blocks")
                if getattr(mlp_block, "activation_function", None) != "swiglu":
                    raise ValueError("attention_group_moe currently requires SwiGLU experts")
            log_rank_0(
                logging.INFO,
                "Attention-group MoE: three depth blocks, one shared and one "
                "column-private expert, "
                f"expert expansion={self.attention_group_moe_expansion_ratio:g}",
            )

        if self.original_input_width:  # fixed_residual_width entails this
            self.embedding_width = self.hidden_size
        else:
            self.embedding_width = self.widths[0]

        if self.original_output_width:  # fixed_residual_width entails this
            self.unembedding_width = self.hidden_size
        else:
            self.unembedding_width = max(self.widths) if self.wide_unembedding else self.widths[-1]

        log_rank_0(logging.INFO, f"WidthVaryingConfig widths per layer: {self.widths}")
        if self.wide_unembedding:
            log_rank_0(
                logging.INFO,
                f"Wide unembedding enabled: final layer={self.widths[-1]}, unembedding={self.unembedding_width}",
            )

        moe_intermediate_size = getattr(self, "moe_intermediate_size", None)
        h = self.hidden_size
        for i, (attn_block, mlp_block) in enumerate(
            zip(self.sequence_mixer_blocks, self.mlp_blocks)
        ):
            width = self.widths[i]
            assert width % attn_block.num_attention_heads == 0
            # using 1/d rather than 1/sqrt(d) due to mup
            attn_block.attention_multiplier = 1 / (width // attn_block.num_attention_heads)
            if getattr(mlp_block, "mlp_type", None) == "MoE":
                assert moe_intermediate_size is not None, (
                    "moe_intermediate_size must be set when using MoE blocks"
                )
                # Constant ratio: intermediate_size / hidden_size per layer
                mlp_block.intermediate_size = int(
                    round(width * moe_intermediate_size / h)
                )
                assert getattr(mlp_block, "shared_intermediate_size", None) in (
                    -1,
                    None,
                ), "moe_shared_intermediate_size must be -1 (no shared expert)"
                mlp_block.shared_intermediate_size = None
            elif self.attention_group_moe:
                mlp_block.intermediate_size = int(
                    round(width * self.attention_group_moe_expansion_ratio)
                )
            else:
                mlp_block.intermediate_size = width * 4
