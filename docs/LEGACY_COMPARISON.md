# Legacy overlap comparison

This note records the intentional differences between the previous
`run_qwen2_5vl_visual_confidence_mathverse.py` experiment and VisConf. The
normative definitions remain the metric documents in this directory.

## Directly comparable probability metrics

For the same raw float32 logit vector and selected token, the following retain
the legacy formulas:

| Legacy field | VisConf field | Expected relationship |
|---|---|---|
| `logp` | `logp` | Equal |
| `gini` | `gini` | Equal Simpson concentration |
| `entropy` | `entropy` | Equal negative Shannon entropy |
| `dist_perplexity` | `dist_perplexity` | Equal negative perplexity |
| `kl` | `kl_u_p` | Equal when every full-vocabulary probability is finite and nonzero |

VisConf computes these metrics from raw model logits before temperature, top-k,
or top-p selection transforms. The legacy script could fall back to processed
generation scores when raw logits were unavailable; those fallback rows are not
comparable.

## Intentional probability differences

- VisConf never reduces the support of `kl_u_p`. If a full-vocabulary
  probability underflows to zero, the mathematically defined result remains
  positive infinity. The legacy fallback instead computed KL on surviving
  processed-score support.
- VisConf stores all 31 documented probability metrics. Metrics without a legacy
  field have no overlap baseline.
- Undefined invalid inputs produce a present metric-family row with null values;
  mathematically defined infinity remains IEEE infinity.

## Attention and hidden-state differences

The old `vis_ratio` and embedding/hidden summaries used different token groups,
layer aggregation, and sometimes post-token rather than predictor-aligned
states. They are not treated as numerical baselines.

VisConf instead:

- aligns logits, attention, and hidden state to the computation predicting the
  stored token;
- uses tokenizer-derived image, prompt-text, and prior-generated groups;
- averages the fixed ranges 1-21 and 22-34 while using decoder layer 36 as the
  actual last layer;
- uses per-layer prompt image prototypes and normalized cosine values;
- reduces eager attention and hidden states online.

These are deliberate experiment-definition changes, not regressions to be
forced into legacy equality.

## Acceptance rule

Real-model smoke acceptance checks exact key alignment and storage schemas,
compares retained token IDs across prompt/rollout batch shapes, and validates the
direct probability overlap above on representative logits. Attention and hidden
state are audited against the current normative definitions only.
