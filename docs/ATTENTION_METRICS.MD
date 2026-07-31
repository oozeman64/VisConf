# Attention Metrics

These metrics are computed at each decoding step from `attention_weights`
on the generated-response side.

The formulas use plain-text notation so they remain readable in Markdown
renderers without LaTeX support.

## Attention aggregation scenarios

Every metric is logged separately for each of these three aggregation
scenarios. Layer numbers are one-based and inclusive:

| Scenario label | Layers | Heads |
| --- | --- | --- |
| `all_layers_all_heads` | 1–36 | All heads |
| `early_visual_integration` | 1–21 | All heads |
| `visual_reasoning` | 22–34 | All heads |

The model has 36 decoder layers. The two phase-specific ranges are fixed
experimental ranges rather than a partition of the full model: layers 35–36
participate in all_layers_all_heads but are intentionally excluded from
early_visual_integration and visual_reasoning.

The metric definitions below are shared across scenarios. The scenario label must
remain attached to each logged metric so values from different aggregation
scenarios are not conflated.

## Attention aggregation

The experiment is predictor-aligned. At decoding step `t`, attention metrics are
collected from the same forward pass whose output distribution predicts `y_t`.
Let `q_t` denote its final query position:

~~~text
q_1 = the final unmasked prompt position used to predict y_1
q_t = the position of y_{t-1},                              t > 1
~~~

For an aggregation scenario `s`, let `L_s` be its selected layer set and let
`H_l` be the number of heads in layer `l`. The attention vector from `q_t` to
all attended key positions is averaged across all selected layer-head pairs:

~~~text
v_t^(s) = (1 / sum_{l in L_s} H_l)
          * sum_{l in L_s} sum_{h=1..H_l}
              Attention^(l,h)(q_t -> ·)
~~~

`Attention^(l,h)` is the post-softmax decoder self-attention probability after
causal and padding masks have been applied. Attention is collected in inference
or evaluation mode with attention dropout disabled. Returned attention weights
are converted to float32 before heads or layers are aggregated. Pre-softmax
attention scores and dropout-modified weights are not used.

For the rest of this document, `v_t` denotes the scenario-specific vector
`v_t^(s)`. Every group metric and cross-group ratio is computed independently
from each of the three scenario-specific vectors.

The eligible source-token positions used by the logged metrics are divided into
three base groups:

- `I`: visual-token positions.
- `P`: eligible prompt-text positions.
- `G`: eligible previously generated text positions.

The groups are defined operationally as follows:

- `G_t = {y_1, ..., y_{t-1}}`, where the `y_i` are retained generated tokens.
  Therefore `G_1` is empty. For `t > 1`, the predictor query `q_t = y_{t-1}`
  is also an attention key and is included in `G_t`; its self-attention is not
  removed. Generated EOS and `<|im_end|>` terminators are trimmed, receive no
  metric row, and do not belong to `G`.
- Ordinary instruction and user text is prompt text. Chat-template role markers
  and other prompt tokens listed in the tokenizer's `all_special_ids` are excluded
  from `P`.
- `<|vision_start|>` and `<|vision_end|>` are excluded. Every position inside an
  image span is excluded from text positions, but only `<|image_pad|>` positions
  belong to `I`. Image-newline and similar positions therefore belong to none of
  `I`, `P`, or `G` and receive no direct group logging.
- BOS and EOS positions are excluded.
- Positions whose attention mask is not `1`, including ordinary padding, are
  excluded from all groups and from all group token counts.

The full attention vector may contain positions that belong to none of these
groups, including excluded special or image-span positions. The first-step query
`q_1` also belongs to none of the groups if that final prompt position is an
excluded special token. Consequently, the `all_attn` group means `I ∪ P ∪ G`,
not every key in the model's attention context, and `all_attn_total` need not
equal `1`.

This makes the first step structurally different when `q_1` is excluded: its
self-attention belongs to no logged group, whereas for `t > 1` the predictor's
self-attention belongs to `G_t`. First-step ratios must therefore be interpreted
using their eligible key set rather than assumed to have exactly the same support
as later-step ratios.

The composite groups are:

- `P ∪ G`: prompt and generated text tokens together.
- `I ∪ P ∪ G`: all image, prompt text, and generated text tokens.

For any group `S`, let `n_S = |S|`, let `v_{S,t}` be the entries of
`v_t` belonging to `S`, and define the group's total attention mass:

~~~text
A_{S,t} = sum_{i in S} v_t[i]
~~~

The group-specific metric names use the following prefixes:

| Token group | Prefix |
| --- | --- |
| Image tokens only | `img_attn` |
| Prompt text tokens only | `prompt_text_attn` |
| Generated text tokens only | `generated_text_attn` |
| Prompt and generated text tokens | `prompt_generated_text_attn` |
| Image, prompt text, and generated text tokens | `all_attn` |

Each prefix receives the seven metrics described below. For example,
`img_attn_total` and `prompt_text_attn_entropy` are per-decoding-step
scalars.

---

## Per-group metrics

### Total attention

For a group `S`:

~~~text
<prefix>_total_t = A_{S,t}
~~~

This is the total attention mass allocated to the group's tokens at step `t`.

### Average attention

~~~text
<prefix>_avg_t = A_{S,t} / n_S
~~~

This is the mean attention per token in the group. It is undefined when the group
contains no tokens.

### Within-group attention distribution

The remaining metrics describe how attention is distributed within the selected
group. When `A_{S,t} >= 1e-8`, normalize the group's attention vector:

~~~text
p_{S,t}[i] = v_t[i] / A_{S,t},    i in S
~~~

If `A_{S,t} < 1e-8`, the five normalized distribution metrics below are
`None`: Gini, entropy, both KL directions, and distribution perplexity.

### Gini concentration

~~~text
<prefix>_gini_t = sum_{i in S} p_{S,t}[i]^2
~~~

This follows the repository's deliberately named `gini` convention: it is Simpson
concentration, also called collision probability. The stored name is intentional.
It is not the conventional Gini coefficient and is not `1 - sum(p_i^2)`.

### Negative entropy

~~~text
<prefix>_entropy_t = sum_{i in S} p_{S,t}[i] * log(p_{S,t}[i])
~~~

This is intentionally negative Shannon entropy despite the stored `entropy`
name. Values closer to zero indicate a more concentrated within-group attention
distribution. Terms with `p_{S,t}[i] = 0` contribute zero under the convention
`0 * log(0) = 0`.

### KL divergence: uniform distribution to attention distribution

Let `U_S` be the uniform distribution over the `n_S` tokens in group
`S`:

~~~text
U_S[i] = 1 / n_S
~~~

~~~text
<prefix>_kl_u_p_t = D_KL(U_S || p_{S,t})
                   = (1 / n_S) * sum_{i in S}
                       log((1 / n_S) / p_{S,t}[i])
~~~

This is the direction used by the earlier `*_attn_kl` draft metric. Zero entries
are not removed and the distribution is not renormalized over positive support.
If any `p_{S,t}[i] = 0`, this KL divergence is `+infinity`. If any required input
is non-finite, the metric is `None`.

### KL divergence: attention distribution to uniform distribution

~~~text
<prefix>_kl_p_u_t = D_KL(p_{S,t} || U_S)
                   = sum_{i in S} p_{S,t}[i]
                       * log(p_{S,t}[i] / (1 / n_S))
~~~

Zero-probability entries contribute zero using the usual `0 * log(0) = 0`
convention. Unlike `D_KL(U_S || p_{S,t})`, this direction does not require
support reduction merely because an entry is zero.

### Negative distribution perplexity

~~~text
<prefix>_dist_perplexity_t
    = -exp(-<prefix>_entropy_t)
~~~

This is negative conventional distribution perplexity. Values closer to `-1`
indicate a more concentrated within-group attention distribution.

### Metric instantiations

The seven metrics are instantiated for all five groups:

- `img_attn_total`, `img_attn_avg`, `img_attn_gini`,
  `img_attn_entropy`, `img_attn_kl_u_p`, `img_attn_kl_p_u`,
  `img_attn_dist_perplexity`
- `prompt_text_attn_total`, `prompt_text_attn_avg`,
  `prompt_text_attn_gini`, `prompt_text_attn_entropy`,
  `prompt_text_attn_kl_u_p`, `prompt_text_attn_kl_p_u`,
  `prompt_text_attn_dist_perplexity`
- `generated_text_attn_total`, `generated_text_attn_avg`,
  `generated_text_attn_gini`, `generated_text_attn_entropy`,
  `generated_text_attn_kl_u_p`, `generated_text_attn_kl_p_u`,
  `generated_text_attn_dist_perplexity`
- `prompt_generated_text_attn_total`,
  `prompt_generated_text_attn_avg`,
  `prompt_generated_text_attn_gini`,
  `prompt_generated_text_attn_entropy`,
  `prompt_generated_text_attn_kl_u_p`,
  `prompt_generated_text_attn_kl_p_u`,
  `prompt_generated_text_attn_dist_perplexity`
- `all_attn_total`, `all_attn_avg`, `all_attn_gini`,
  `all_attn_entropy`, `all_attn_kl_u_p`, `all_attn_kl_p_u`,
  `all_attn_dist_perplexity`

If existing code exposes `img_attn_kl`, it is the legacy name for
`img_attn_kl_u_p`.

---

## Cross-group attention ratios

The ratio metrics use the raw total attention masses for the three base groups:

~~~text
A_img,t      = image-token attention mass
A_prompt,t   = prompt-text attention mass
A_gen,t      = generated-text attention mass
A_all,t      = A_img,t + A_prompt,t + A_gen,t
~~~

They are computed from totals, not from the normalized within-group distributions.
When a ratio denominator is smaller than `1e-8`, the ratio is `None`.
Otherwise, valid ratios lie in the range `[0, 1]`.

### Image attention relative to prompt text

~~~text
attn_ratio_img_to_prompt_text_t
    = A_img,t / (A_img,t + A_prompt,t)
~~~

Fraction of image-plus-prompt attention assigned to image tokens.

### Image attention relative to generated text

~~~text
attn_ratio_img_to_generated_text_t
    = A_img,t / (A_img,t + A_gen,t)
~~~

Fraction of image-plus-generated-text attention assigned to image tokens.

### Image attention relative to all attention

~~~text
attn_ratio_img_to_all_t
    = A_img,t / (A_img,t + A_prompt,t + A_gen,t)
    = A_img,t / A_all,t
~~~

Fraction of total image-plus-text attention assigned to image tokens.

### Prompt text attention relative to all attention

~~~text
attn_ratio_prompt_text_to_all_t
    = A_prompt,t / (A_img,t + A_prompt,t + A_gen,t)
    = A_prompt,t / A_all,t
~~~

Fraction of total image-plus-text attention assigned to prompt text tokens.

### Generated text attention relative to all attention

~~~text
attn_ratio_generated_text_to_all_t
    = A_gen,t / (A_img,t + A_prompt,t + A_gen,t)
    = A_gen,t / A_all,t
~~~

Fraction of total image-plus-text attention assigned to generated text tokens.

### Prompt and generated text attention relative to all attention

~~~text
attn_ratio_prompt_generated_text_to_all_t
    = (A_prompt,t + A_gen,t)
      / (A_img,t + A_prompt,t + A_gen,t)
    = (A_prompt,t + A_gen,t) / A_all,t
~~~

Fraction of total image-plus-text attention assigned to text tokens of either
kind.

The ratios have these useful consistency relationships:

~~~text
attn_ratio_img_to_all_t
    + attn_ratio_prompt_generated_text_to_all_t
    = 1

attn_ratio_prompt_text_to_all_t
    + attn_ratio_generated_text_to_all_t
    = attn_ratio_prompt_generated_text_to_all_t
~~~

---

## Edge cases and conventions

- Before aggregation, every attention value in the selected `q_t` query rows
  across the scenario's layers, heads, and key positions must be finite. If any
  such value is NaN or infinite, every attention metric for that aggregation
  scenario at that decoding step is `None`.
- Values in the interval `[-1e-8, 0)` are treated as floating-point excursions
  and clipped to zero before aggregation. Any value smaller than `-1e-8`
  invalidates every attention metric for that aggregation scenario at that step.
- The complete attention vector is not renormalized after validation or clipping.
  Only the explicitly defined within-group distributions are normalized.
- If a token group is empty, its total attention is zero, its average is
  undefined, and its normalized distribution metrics are `None`.
- The `1e-8` threshold applies to normalized group metrics and ratio
  denominators.
- Natural logarithms are used.
- Terms of the form `0 * log(0)` are defined as zero.
- The total metric is a mass or fraction from the original attention vector;
  it is not renormalized across groups.
- The average metric divides by the number of tokens in the selected group, not
  by the total vocabulary or context length.
