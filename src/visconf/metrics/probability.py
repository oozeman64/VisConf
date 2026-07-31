"""Predictor-aligned token-probability confidence metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from visconf.metrics.validation import probability_logits


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


@torch.inference_mode()
def compute_probability_metrics(
    raw_logits: torch.Tensor,
    selected_token_id: int,
) -> ProbabilityMetrics:
    """Compute all 31 probability metrics from the unmodified raw logits."""

    logits = probability_logits(raw_logits, selected_token_id)
    vocabulary_size = logits.numel()
    log_vocabulary_size = math.log(vocabulary_size)

    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = torch.exp(log_probabilities)
    sorted_log_probabilities, _ = torch.sort(log_probabilities, descending=True)
    sorted_probabilities = torch.exp(sorted_log_probabilities)

    selected_logp = log_probabilities[selected_token_id]
    entropy = _xlog_from_log_probability(
        probabilities, log_probabilities
    ).sum()
    gini = torch.exp(2 * log_probabilities).sum()

    other_logits = log_probabilities.clone()
    other_logits[selected_token_id] = -torch.inf
    selected_dominance = selected_logp - other_logits.max()
    competition_rank = 1 + int(
        torch.count_nonzero(log_probabilities > selected_logp).item()
    )

    topk_mass: dict[int, torch.Tensor] = {}
    for k in (2, 5, 10, 20):
        topk_mass[k] = sorted_probabilities[: min(k, vocabulary_size)].sum()

    log_uniform = -log_vocabulary_size
    log_mixture = torch.logaddexp(
        log_probabilities,
        torch.full_like(log_probabilities, log_uniform),
    ) - math.log(2)
    kl_p_m = torch.where(
        probabilities > 0,
        probabilities * (log_probabilities - log_mixture),
        torch.zeros_like(probabilities),
    ).sum()
    kl_u_m = (log_uniform - log_mixture).mean()
    js_distance = torch.sqrt(torch.clamp(0.5 * (kl_p_m + kl_u_m), min=0))

    renyi_half = (
        torch.logsumexp(0.5 * log_probabilities, dim=0) / (0.5 - 1)
    )
    renyi_two = torch.logsumexp(2 * log_probabilities, dim=0)
    renyi_four = torch.logsumexp(4 * log_probabilities, dim=0) / 3

    return ProbabilityMetrics(
        logp=float(selected_logp.item()),
        kl_u_p=float(
            (-log_vocabulary_size - log_probabilities.mean()).item()
        ),
        kl_p_u=float((log_vocabulary_size + entropy).item()),
        gini=float(gini.item()),
        entropy=float(entropy.item()),
        dist_perplexity=float((-torch.exp(-entropy)).item()),
        inverse_perplexity=float(torch.exp(entropy).item()),
        max_prob=float(sorted_probabilities[0].item()),
        margin_top2=float(
            (sorted_probabilities[0] - sorted_probabilities[1]).item()
        ),
        log_ratio_margin_top2=float(
            (sorted_log_probabilities[0] - sorted_log_probabilities[1]).item()
        ),
        selected_dominance=float(selected_dominance.item()),
        selected_logrank=-math.log(competition_rank),
        topk_mass_2=float(topk_mass[2].item()),
        topk_mass_5=float(topk_mass[5].item()),
        topk_mass_10=float(topk_mass[10].item()),
        topk_mass_20=float(topk_mass[20].item()),
        tail_mass_2=float((topk_mass[2] - 1).item()),
        tail_mass_5=float((topk_mass[5] - 1).item()),
        tail_mass_10=float((topk_mass[10] - 1).item()),
        tail_mass_20=float((topk_mass[20] - 1).item()),
        norm_entropy_concentration=float(
            (1 + entropy / log_vocabulary_size).item()
        ),
        renyi_entropy_0p5=float(renyi_half.item()),
        renyi_entropy_1=float(entropy.item()),
        renyi_entropy_2=float(renyi_two.item()),
        renyi_entropy_4=float(renyi_four.item()),
        renyi_entropy_inf=float(sorted_log_probabilities[0].item()),
        js_p_u=float(js_distance.item()),
        selected_rank=-competition_rank,
        nucleus_size_0p9=_nucleus_size(sorted_probabilities, 0.90),
        nucleus_size_0p95=_nucleus_size(sorted_probabilities, 0.95),
        nucleus_size_0p99=_nucleus_size(sorted_probabilities, 0.99),
    )
