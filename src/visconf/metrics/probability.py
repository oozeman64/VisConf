"""Predictor-aligned token-probability confidence metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from visconf.metrics.validation import (
    probability_logits,
    probability_logits_batch,
)


PROBABILITY_FLOAT_METRICS = (
    "logp",
    "kl_u_p",
    "kl_p_u",
    "gini",
    "entropy",
    "dist_perplexity",
    "inverse_perplexity",
    "max_prob",
    "margin_top2",
    "log_ratio_margin_top2",
    "selected_dominance",
    "selected_logrank",
    "topk_mass_2",
    "topk_mass_5",
    "topk_mass_10",
    "topk_mass_20",
    "tail_mass_2",
    "tail_mass_5",
    "tail_mass_10",
    "tail_mass_20",
    "norm_entropy_concentration",
    "renyi_entropy_0p5",
    "renyi_entropy_1",
    "renyi_entropy_2",
    "renyi_entropy_4",
    "renyi_entropy_inf",
    "js_p_u",
)
PROBABILITY_INTEGER_METRICS = (
    "selected_rank",
    "nucleus_size_0p9",
    "nucleus_size_0p95",
    "nucleus_size_0p99",
)
PROBABILITY_METRICS = PROBABILITY_FLOAT_METRICS + PROBABILITY_INTEGER_METRICS


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    logp: float
    kl_u_p: float
    kl_p_u: float
    gini: float
    entropy: float
    dist_perplexity: float
    inverse_perplexity: float
    max_prob: float
    margin_top2: float
    log_ratio_margin_top2: float
    selected_dominance: float
    selected_logrank: float
    topk_mass_2: float
    topk_mass_5: float
    topk_mass_10: float
    topk_mass_20: float
    tail_mass_2: float
    tail_mass_5: float
    tail_mass_10: float
    tail_mass_20: float
    norm_entropy_concentration: float
    renyi_entropy_0p5: float
    renyi_entropy_1: float
    renyi_entropy_2: float
    renyi_entropy_4: float
    renyi_entropy_inf: float
    js_p_u: float
    selected_rank: int
    nucleus_size_0p9: int
    nucleus_size_0p95: int
    nucleus_size_0p99: int


def _xlog_from_log_probability(
    probabilities: torch.Tensor,
    log_probabilities: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        probabilities > 0,
        probabilities * log_probabilities,
        torch.zeros_like(probabilities),
    )


def _nucleus_size(sorted_probabilities: torch.Tensor, threshold: float) -> int:
    cumulative = torch.cumsum(sorted_probabilities, dim=0)
    crossing = torch.nonzero(cumulative >= threshold, as_tuple=False)
    size = (
        int(crossing[0, 0].item()) + 1
        if crossing.numel()
        else sorted_probabilities.numel()
    )
    return -size


def _nucleus_sizes(
    sorted_probabilities: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    crossing = cumulative >= threshold
    first = crossing.to(dtype=torch.int64).argmax(dim=-1) + 1
    fallback = torch.full_like(first, sorted_probabilities.shape[1])
    return -torch.where(crossing.any(dim=-1), first, fallback)


@torch.inference_mode()
def compute_probability_metrics_batch(
    raw_logits: torch.Tensor,
    selected_token_ids: torch.Tensor,
) -> tuple[ProbabilityMetrics, ...]:
    """Compute all probability metrics for a logits microbatch."""

    logits, selected = probability_logits_batch(
        raw_logits,
        selected_token_ids,
    )
    batch_size, vocabulary_size = logits.shape
    log_vocabulary_size = math.log(vocabulary_size)

    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = torch.exp(log_probabilities)
    sorted_log_probabilities, _ = torch.sort(
        log_probabilities,
        dim=-1,
        descending=True,
    )
    sorted_probabilities = torch.exp(sorted_log_probabilities)
    selected_logp = log_probabilities.gather(
        1,
        selected.unsqueeze(1),
    ).squeeze(1)
    entropy = _xlog_from_log_probability(
        probabilities,
        log_probabilities,
    ).sum(dim=-1)
    gini = torch.exp(2 * log_probabilities).sum(dim=-1)

    other_logits = log_probabilities.clone()
    other_logits.scatter_(1, selected.unsqueeze(1), -torch.inf)
    selected_dominance = selected_logp - other_logits.max(dim=-1).values
    competition_rank = 1 + torch.count_nonzero(
        log_probabilities > selected_logp.unsqueeze(1),
        dim=-1,
    )

    topk_mass = {
        k: sorted_probabilities[:, : min(k, vocabulary_size)].sum(dim=-1)
        for k in (2, 5, 10, 20)
    }
    log_uniform = -log_vocabulary_size
    log_mixture = torch.logaddexp(
        log_probabilities,
        torch.full_like(log_probabilities, log_uniform),
    ) - math.log(2)
    kl_p_m = torch.where(
        probabilities > 0,
        probabilities * (log_probabilities - log_mixture),
        torch.zeros_like(probabilities),
    ).sum(dim=-1)
    kl_u_m = (log_uniform - log_mixture).mean(dim=-1)
    js_distance = torch.sqrt(torch.clamp(0.5 * (kl_p_m + kl_u_m), min=0))

    renyi_half = (
        torch.logsumexp(0.5 * log_probabilities, dim=-1) / (0.5 - 1)
    )
    renyi_two = torch.logsumexp(2 * log_probabilities, dim=-1)
    renyi_four = torch.logsumexp(4 * log_probabilities, dim=-1) / 3

    float_values = torch.stack(
        (
            selected_logp,
            -log_vocabulary_size - log_probabilities.mean(dim=-1),
            log_vocabulary_size + entropy,
            gini,
            entropy,
            -torch.exp(-entropy),
            torch.exp(entropy),
            sorted_probabilities[:, 0],
            sorted_probabilities[:, 0] - sorted_probabilities[:, 1],
            sorted_log_probabilities[:, 0] - sorted_log_probabilities[:, 1],
            selected_dominance,
            -torch.log(competition_rank.to(dtype=torch.float32)),
            topk_mass[2],
            topk_mass[5],
            topk_mass[10],
            topk_mass[20],
            topk_mass[2] - 1,
            topk_mass[5] - 1,
            topk_mass[10] - 1,
            topk_mass[20] - 1,
            1 + entropy / log_vocabulary_size,
            renyi_half,
            entropy,
            renyi_two,
            renyi_four,
            sorted_log_probabilities[:, 0],
            js_distance,
        ),
        dim=1,
    ).cpu().tolist()
    integer_values = torch.stack(
        (
            -competition_rank,
            _nucleus_sizes(sorted_probabilities, 0.90),
            _nucleus_sizes(sorted_probabilities, 0.95),
            _nucleus_sizes(sorted_probabilities, 0.99),
        ),
        dim=1,
    ).cpu().tolist()
    return tuple(
        ProbabilityMetrics(
            *float_values[index],
            *integer_values[index],
        )
        for index in range(batch_size)
    )


@torch.inference_mode()
def compute_probability_metrics(
    raw_logits: torch.Tensor,
    selected_token_id: int,
) -> ProbabilityMetrics:
    """Compute all 31 probability metrics from the unmodified raw logits."""

    probability_logits(raw_logits, selected_token_id)
    return compute_probability_metrics_batch(
        raw_logits.unsqueeze(0),
        torch.tensor(
            [selected_token_id],
            dtype=torch.long,
            device=raw_logits.device,
        ),
    )[0]
