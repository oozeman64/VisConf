# Token-Probability Confidence Metrics

All metrics are computed at decoding step `t` from the next-token distribution
that predicts `y_t`:

~~~text
p_t(j) = P(j | x, y_<t),    j ∈ {1, ..., V}
~~~

where `V = output.logits.shape[-1]` is the full width of the model's raw-logit
vector and `y_t` is the generated token. All `V` coordinates participate in
`p_t`, including special-token, EOS, and `<|im_end|>` coordinates. EOS and
`<|im_end|>` remain candidates in distributions for retained steps even though a
row in which either terminator is selected is itself trimmed. Let

~~~text
p_t,(1) ≥ p_t,(2) ≥ ... ≥ p_t,(V)
~~~

denote the probabilities sorted in descending order, and let `U(j) = 1/V`
be the uniform distribution.

The experiment is predictor-aligned across metric families. Probability,
attention, and hidden-state metrics for step `t` all come from the forward pass
that predicts `y_t` from `x, y_<t`. Thus, all metrics in a step describe the
same predictive computation rather than the representation obtained after
processing `y_t`.
Generated EOS and `<|im_end|>` terminators are trimmed from the retained generated
sequence, so no metric row is stored for either terminator.

`p_t` is computed strictly from the model's raw `output.logits` at the position
that predicts `y_t`. No temperature scaling, sampling transformation, top-k or
top-p truncation, repetition penalty, or other decoding processor is applied
before computing `p_t`. The decoding procedure used to select `y_t` does not
alter the distribution used by these metrics. If raw `output.logits` are not
available, metric collection fails explicitly; processed generation scores must
not be substituted.

All probability computations and reductions use float32 or higher precision.
Define

~~~text
log_p_t = log_softmax(raw_logits_t.float(), dim=-1)
p_t     = exp(log_p_t)
~~~

Metrics are evaluated with numerically stable log-domain expressions whenever
available. In particular,

~~~text
kl_u_p_t = -log(V) - (1 / V) * sum_{j=1..V} log_p_t(j)
~~~

No zero-probability or underflowed entry is removed from a metric's support.
Mathematically defined positive or negative infinity is preserved as infinity;
it is not converted to `None` merely because it is non-finite.

> Formulas use plain-text notation so they remain readable in Markdown renderers
> without LaTeX support.

**Convention:** larger values always indicate greater confidence. Metrics that are
conventionally smaller for more confident distributions are intentionally
sign-inverted or otherwise transformed below. In particular, metrics named
`entropy` and `renyi_entropy_*` store negative entropy by design; their names are
retained intentionally. Natural logarithms are assumed. Terms of the form
`0 * log(0)` are defined as zero.

---

## `logp`

~~~text
logp_t = log(p_t(y_t))
~~~

Log-probability assigned to the generated token. Higher values, meaning values
closer to `0`, indicate that the model assigned more probability to the
selected token.

---

## `kl_u_p`

~~~text
kl_u_p_t = D_KL(U || p_t)
          = (1 / V) * sum_{j=1..V} log((1 / V) / p_t(j))
~~~

Measures how far the token distribution is from uniform, with the uniform
distribution as the reference distribution. Higher values indicate greater
concentration and therefore greater confidence. This metric is highly sensitive
to very small tail probabilities and is infinite if any `p_t(j) = 0`.

---

## `kl_p_u`

~~~text
kl_p_u_t = D_KL(p_t || U)
          = sum_{j=1..V} p_t(j) * log(p_t(j) / (1 / V))
~~~

Equivalently,

~~~text
kl_p_u_t = log(V) + sum_{j=1..V} p_t(j) * log(p_t(j))
~~~

Measures how concentrated the token distribution is relative to a uniform
distribution. Higher values indicate greater confidence. It is an affine
transformation of negative Shannon entropy.

---

## `gini`

~~~text
gini_t = sum_{j=1..V} p_t(j)^2
~~~

Simpson concentration, also called collision probability. Higher values indicate
that probability mass is concentrated on fewer tokens. This is `1` minus
the conventional Gini impurity. The stored name `gini` is retained intentionally
for this Simpson-concentration convention; it does not denote the conventional
Gini coefficient.

---

## `entropy`

~~~text
entropy_t = sum_{j=1..V} p_t(j) * log(p_t(j))
~~~

Negative Shannon entropy. Conventional entropy is lower for more confident
distributions, so its sign is inverted here. Higher values, meaning values
closer to `0`, indicate greater confidence.

---

## `dist_perplexity`

~~~text
dist_perplexity_t = -exp(
    -sum_{j=1..V} p_t(j) * log(p_t(j))
)
~~~

Negative distribution perplexity. Conventional distribution perplexity is lower
for more confident distributions, so its sign is inverted here. Higher values,
meaning values closer to `-1`, indicate greater confidence.

---

## `inverse_perplexity`

~~~text
inverse_perplexity_t = exp(
    sum_{j=1..V} p_t(j) * log(p_t(j))
)
~~~

The reciprocal of conventional distribution perplexity. Higher values indicate
greater confidence. Its range is `[1/V, 1]`, with `1` corresponding
to a point-mass distribution.

---

## `max_prob`

~~~text
max_prob_t = p_t,(1)
~~~

Probability assigned to the most likely next token. Higher values indicate a
more dominant modal prediction.

---

## `margin_top2`

~~~text
margin_top2_t = p_t,(1) - p_t,(2)
~~~

Difference between the largest and second-largest token probabilities. Higher
values indicate clearer separation between the two strongest alternatives.

---

## `log_ratio_margin_top2`

~~~text
log_ratio_margin_top2_t
    = log(p_t,(1)) - log(p_t,(2))
    = log(p_t,(1) / p_t,(2))
~~~

Log-probability ratio between the most likely and second-most-likely tokens.
Higher values indicate stronger relative preference for the top token. The value
is infinite when `p_t,(2) = 0`.

---

## `selected_dominance`

~~~text
selected_dominance_t
    = log(p_t(y_t)) - max_{j != y_t} log(p_t(j))
~~~

Log-probability advantage of the generated token over its strongest alternative.
Higher values indicate that the selected token is more dominant. The value is
positive when the generated token is the unique modal token and negative when
decoding selects a non-modal token.

---

## `selected_rank`

Let `r_t ∈ {1, ..., V}` be the competition rank of `y_t`, with rank
`1` being best:

~~~text
r_t = 1 + |{j : p_t(j) > p_t(y_t)}|
~~~

Tokens tied with `y_t` receive the same rank. Thus, a token tied for the highest
probability has rank `1`.

~~~text
selected_rank_t = -r_t
~~~

The conventional rank is lower for more confident selections, so its sign is
inverted here. Higher values, meaning values closer to `-1`, indicate that
the generated token was ranked more highly.

---

## `selected_logrank`

~~~text
selected_logrank_t = -log(r_t)
~~~

Negative log-rank of the generated token. Conventional log-rank is lower for
more confident selections, so its sign is inverted here. Higher values, meaning
values closer to `0`, indicate greater confidence.

---

## `topk_mass`

For `k ∈ {2, 5, 10, 20}`,

~~~text
topk_mass_t,k = sum_{i=1..k} p_t,(i)
~~~

Probability mass assigned to the `k` most likely tokens. Higher values
indicate that more of the distribution is concentrated within a small candidate
set.

Instantiations:

- `topk_mass_2`
- `topk_mass_5`
- `topk_mass_10`
- `topk_mass_20`

---

## `tail_mass`

For `k ∈ {2, 5, 10, 20}`, define the conventional tail mass as

~~~text
T_t,k = sum_{i=k+1..V} p_t,(i)
      = 1 - sum_{i=1..k} p_t,(i)
~~~

Because conventional tail mass is lower for more confident distributions, it is
sign-inverted:

~~~text
tail_mass_t,k = -T_t,k
              = sum_{i=1..k} p_t,(i) - 1
~~~

Higher values, meaning values closer to `0`, indicate greater confidence. For
a fixed `k`, this metric is an affine transformation of `topk_mass`.

Instantiations:

- `tail_mass_2`
- `tail_mass_5`
- `tail_mass_10`
- `tail_mass_20`

---

## `nucleus_size`

For `q ∈ {0.90, 0.95, 0.99}`, let

~~~text
K_t,q = min { k : sum_{i=1..k} p_t,(i) >= q }
~~~

Conventional nucleus size is lower for more confident distributions, so its sign
is inverted:

~~~text
nucleus_size_t,q = -K_t,q
~~~

Higher values, meaning values closer to `-1`, indicate that fewer tokens are
needed to account for the specified cumulative probability mass.

Instantiations:

- `nucleus_size_0.9`
- `nucleus_size_0.95`
- `nucleus_size_0.99`

---

## `norm_entropy_concentration`

~~~text
norm_entropy_concentration_t
    = 1 - ( -sum_{j=1..V} p_t(j) * log(p_t(j)) ) / log(V)
~~~

One minus Shannon entropy normalized by the maximum possible entropy. Higher
values indicate greater concentration. Its range is `[0, 1]`, where `0`
corresponds to a uniform distribution and `1` to a point mass.

---

## `renyi_entropy`

For `α ∈ {0.5, 1, 2, 4, ∞}`, conventional Rényi entropy is lower for more
confident distributions, so its sign is inverted.

For `α != 1, ∞`,

~~~text
renyi_entropy_t,α
    = (1 / (α - 1)) * log(sum_{j=1..V} p_t(j)^α)
~~~

For `α = 1`,

~~~text
renyi_entropy_t,1 = sum_{j=1..V} p_t(j) * log(p_t(j))
~~~

For `α = ∞`,

~~~text
renyi_entropy_t,∞ = log(p_t,(1))
~~~

Higher values indicate greater concentration. Smaller `α` values are more
sensitive to the probability tail, while larger `α` values place more
emphasis on the largest probabilities.

Instantiations:

- `renyi_entropy_0.5`
- `renyi_entropy_1`
- `renyi_entropy_2`
- `renyi_entropy_4`
- `renyi_entropy_inf`

Notable equivalences:

- `renyi_entropy_1` equals `entropy`.
- `renyi_entropy_2` equals `log(gini)`.
- `renyi_entropy_inf` equals `log(max_prob)`.

---

## `js_p_u`

Let

~~~text
M_t = (p_t + U) / 2
~~~

Define the Jensen–Shannon divergence as

~~~text
D_JS(p_t, U)
    = 0.5 * D_KL(p_t || M_t)
    + 0.5 * D_KL(U || M_t)
~~~

The Jensen–Shannon distance is

~~~text
js_p_u_t = sqrt(D_JS(p_t, U))
~~~

Higher values indicate that the token distribution is farther from uniform and
therefore more concentrated. No inversion is required. `js_p_u` is strictly the
Jensen–Shannon distance shown above, including the square root; Jensen–Shannon
divergence is not stored under this name.

---
