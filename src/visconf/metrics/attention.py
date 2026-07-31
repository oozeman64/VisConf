"""Attention metrics over one scenario-averaged predictor query vector."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from visconf.metrics.validation import (
    ATTENTION_EPSILON,
    MetricInputError,
    attention_vector,
)


ATTENTION_SCENARIOS = (
    "all_layers_all_heads",
    "early_visual_integration",
    "visual_reasoning",
)
ATTENTION_SCENARIO_LAYERS = {
    "all_layers_all_heads": frozenset(range(1, 37)),
    "early_visual_integration": frozenset(range(1, 22)),
    "visual_reasoning": frozenset(range(22, 35)),
}
ATTENTION_GROUP_PREFIXES = (
    "img_attn",
    "prompt_text_attn",
    "generated_text_attn",
    "prompt_generated_text_attn",
    "all_attn",
)
ATTENTION_GROUP_SUFFIXES = (
    "total",
    "avg",
    "gini",
    "entropy",
    "kl_u_p",
    "kl_p_u",
    "dist_perplexity",
)
ATTENTION_RATIO_METRICS = (
    "attn_ratio_img_to_prompt_text",
    "attn_ratio_img_to_generated_text",
    "attn_ratio_img_to_all",
    "attn_ratio_prompt_text_to_all",
    "attn_ratio_generated_text_to_all",
    "attn_ratio_prompt_generated_text_to_all",
)
ATTENTION_METRICS = tuple(
    f"{prefix}_{suffix}"
    for prefix in ATTENTION_GROUP_PREFIXES
    for suffix in ATTENTION_GROUP_SUFFIXES
) + ATTENTION_RATIO_METRICS


@dataclass(frozen=True, slots=True)
class StepTokenGroups:
    image_positions: tuple[int, ...]
    prompt_text_positions: tuple[int, ...]
    generated_text_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AttentionScenarioMetrics:
    valid: bool
    img_attn_total: float | None
    img_attn_avg: float | None
    img_attn_gini: float | None
    img_attn_entropy: float | None
    img_attn_kl_u_p: float | None
    img_attn_kl_p_u: float | None
    img_attn_dist_perplexity: float | None
    prompt_text_attn_total: float | None
    prompt_text_attn_avg: float | None
    prompt_text_attn_gini: float | None
    prompt_text_attn_entropy: float | None
    prompt_text_attn_kl_u_p: float | None
    prompt_text_attn_kl_p_u: float | None
    prompt_text_attn_dist_perplexity: float | None
    generated_text_attn_total: float | None
    generated_text_attn_avg: float | None
    generated_text_attn_gini: float | None
    generated_text_attn_entropy: float | None
    generated_text_attn_kl_u_p: float | None
    generated_text_attn_kl_p_u: float | None
    generated_text_attn_dist_perplexity: float | None
    prompt_generated_text_attn_total: float | None
    prompt_generated_text_attn_avg: float | None
    prompt_generated_text_attn_gini: float | None
    prompt_generated_text_attn_entropy: float | None
    prompt_generated_text_attn_kl_u_p: float | None
    prompt_generated_text_attn_kl_p_u: float | None
    prompt_generated_text_attn_dist_perplexity: float | None
    all_attn_total: float | None
    all_attn_avg: float | None
    all_attn_gini: float | None
    all_attn_entropy: float | None
    all_attn_kl_u_p: float | None
    all_attn_kl_p_u: float | None
    all_attn_dist_perplexity: float | None
    attn_ratio_img_to_prompt_text: float | None
    attn_ratio_img_to_generated_text: float | None
    attn_ratio_img_to_all: float | None
    attn_ratio_prompt_text_to_all: float | None
    attn_ratio_generated_text_to_all: float | None
    attn_ratio_prompt_generated_text_to_all: float | None


def _validate_groups(groups: StepTokenGroups, vector_length: int) -> None:
    position_groups = (
        groups.image_positions,
        groups.prompt_text_positions,
        groups.generated_text_positions,
    )
    flattened = [position for group in position_groups for position in group]
    if any(position < 0 or position >= vector_length for position in flattened):
        raise MetricInputError("token-group position is outside the attention vector")
    if any(len(group) != len(set(group)) for group in position_groups):
        raise MetricInputError("token-group positions must be unique")
    if len(flattened) != len(set(flattened)):
        raise MetricInputError("token groups must be disjoint")


def _group_metrics(
    vector: torch.Tensor,
    positions: tuple[int, ...],
    prefix: str,
) -> tuple[dict[str, float | None], float]:
    if not positions:
        return {
            f"{prefix}_total": 0.0,
            f"{prefix}_avg": None,
            f"{prefix}_gini": None,
            f"{prefix}_entropy": None,
            f"{prefix}_kl_u_p": None,
            f"{prefix}_kl_p_u": None,
            f"{prefix}_dist_perplexity": None,
        }, 0.0

    indices = torch.tensor(positions, dtype=torch.long, device=vector.device)
    values = vector.index_select(0, indices)
    total = values.sum()
    total_value = float(total.item())
    metrics: dict[str, float | None] = {
        f"{prefix}_total": total_value,
        f"{prefix}_avg": total_value / len(positions),
    }

    if total_value < ATTENTION_EPSILON:
        metrics.update(
            {
                f"{prefix}_gini": None,
                f"{prefix}_entropy": None,
                f"{prefix}_kl_u_p": None,
                f"{prefix}_kl_p_u": None,
                f"{prefix}_dist_perplexity": None,
            }
        )
        return metrics, total_value

    distribution = values / total
    positive = distribution > 0
    log_distribution = torch.full_like(distribution, -torch.inf)
    log_distribution[positive] = torch.log(distribution[positive])
    entropy = torch.where(
        positive,
        distribution * log_distribution,
        torch.zeros_like(distribution),
    ).sum()
    kl_u_p = (
        torch.tensor(torch.inf, device=vector.device)
        if not torch.all(positive)
        else -math.log(len(positions)) - log_distribution.mean()
    )
    kl_p_u = math.log(len(positions)) + entropy

    metrics.update(
        {
            f"{prefix}_gini": float(torch.sum(distribution.square()).item()),
            f"{prefix}_entropy": float(entropy.item()),
            f"{prefix}_kl_u_p": float(kl_u_p.item()),
            f"{prefix}_kl_p_u": float(kl_p_u.item()),
            f"{prefix}_dist_perplexity": float((-torch.exp(-entropy)).item()),
        }
    )
    return metrics, total_value


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator < ATTENTION_EPSILON else numerator / denominator


@torch.inference_mode()
def compute_attention_metrics(
    scenario_vector: torch.Tensor,
    groups: StepTokenGroups,
) -> AttentionScenarioMetrics:
    """Compute the 41 documented attention metrics for one scenario."""

    vector, _ = attention_vector(scenario_vector)
    _validate_groups(groups, scenario_vector.numel())
    if vector is None:
        return AttentionScenarioMetrics(
            valid=False,
            **{name: None for name in ATTENTION_METRICS},
        )

    image = groups.image_positions
    prompt = groups.prompt_text_positions
    generated = groups.generated_text_positions
    prompt_generated = prompt + generated
    all_positions = image + prompt_generated

    values: dict[str, float | None] = {}
    totals: dict[str, float] = {}
    for prefix, positions in (
        ("img_attn", image),
        ("prompt_text_attn", prompt),
        ("generated_text_attn", generated),
        ("prompt_generated_text_attn", prompt_generated),
        ("all_attn", all_positions),
    ):
        group_values, total = _group_metrics(vector, positions, prefix)
        values.update(group_values)
        totals[prefix] = total

    image_total = totals["img_attn"]
    prompt_total = totals["prompt_text_attn"]
    generated_total = totals["generated_text_attn"]
    all_total = image_total + prompt_total + generated_total

    values.update(
        {
            "attn_ratio_img_to_prompt_text": _ratio(
                image_total, image_total + prompt_total
            ),
            "attn_ratio_img_to_generated_text": _ratio(
                image_total, image_total + generated_total
            ),
            "attn_ratio_img_to_all": _ratio(image_total, all_total),
            "attn_ratio_prompt_text_to_all": _ratio(prompt_total, all_total),
            "attn_ratio_generated_text_to_all": _ratio(
                generated_total, all_total
            ),
            "attn_ratio_prompt_generated_text_to_all": _ratio(
                prompt_total + generated_total, all_total
            ),
        }
    )
    return AttentionScenarioMetrics(valid=True, **values)
