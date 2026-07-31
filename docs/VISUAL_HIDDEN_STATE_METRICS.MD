# Visual Hidden-State Metrics

This document defines one metric family that measures the similarity between the
hidden state used to predict a generated token and an image-token hidden-state
prototype. The metric is computed at every decoding step, but only four
aggregated variants are stored.

The formulas use plain-text notation so they remain readable in Markdown
renderers without LaTeX support.

## Per-layer cosine similarity

The experiment is predictor-aligned. These metrics come from the same forward
pass whose probability distribution predicts `y_t`. Let `q_t` be the predictor
query position defined in the attention-metrics document: the final unmasked
prompt position for `t = 1`, and the position of `y_{t-1}` for `t > 1`.

Layer numbers are one-based and range from 1 through 36. The hidden state at
layer `l` means the output of transformer layer `l`, not its input or the
embedding state. It includes all operations internal to that layer but is taken
before any separate model-level final normalization applied after layer 36. At
decoding step `t`, let `x_t^l` be this output hidden state at `q_t`. Thus
`x_t^l` is the layer-`l` state participating in the computation that predicts
`y_t`, not the state obtained after processing `y_t`. Let `I` be the set of
`<|image_pad|>` visual-token positions defined by the attention-token grouping,
let `n_img = |I|`, and let `h_i^l` be the output hidden state of visual token
`i` at layer `l`.

The image prototype at layer `l` is the mean-pooled hidden state of the image
tokens:

~~~text
μ^l = (1 / n_img) * sum_{i in I} h_i^l
~~~

The layer-specific image-token states and `μ^l` are captured once during the
initial full-prompt forward pass and reused for every decoding step. They are
retained in float32 or higher precision; they are not quantized to float16 for
storage between steps.

The raw cosine similarity between the predictor hidden state and the image
prototype is:

~~~text
cos_raw_t,l
    = (x_t^l · μ^l) / (||x_t^l||_2 * ||μ^l||_2)
~~~

The stored per-layer value is normalized from the cosine range `[-1, 1]` to
`[0, 1]`:

~~~text
cosine_gen_imgproto_hidden_layer_l_t
    = (cos_raw_t,l + 1) / 2
~~~

After normalization:

- `0` corresponds to opposite directions.
- `0.5` corresponds to orthogonal hidden states.
- `1` corresponds to identical directions.

The per-layer value is an intermediate value used to compute the four stored
variants below; the individual layer values are not stored as separate metrics.
The `gen` component in the stored metric names refers to the generated token being
predicted; for `t = 1`, the measured predictor position itself is still a prompt
position.

---

## Stored variants

Each variant is one scalar per decoding step.

| Stored metric | Layer aggregation |
| --- | --- |
| `cosine_gen_imgproto_hidden_avg_all_layers` | Average of layers 1–36 |
| `cosine_gen_imgproto_hidden_last_layer` | Layer 36 only |
| `cosine_gen_imgproto_hidden_early_visual_integration` | Average of layers 1–21 |
| `cosine_gen_imgproto_hidden_visual_reasoning` | Average of layers 22–34 |

The two phase-specific ranges are fixed experimental ranges rather than a
partition of all decoder layers. Layers 35–36 contribute to the all-layer
average, and layer 36 supplies the last-layer metric, but neither layer is
included in the early-visual-integration or visual-reasoning averages.

### Average across all layers

~~~text
cosine_gen_imgproto_hidden_avg_all_layers_t
    = (1 / |V_all,t|) * sum_{l in V_all,t}
        cosine_gen_imgproto_hidden_layer_l_t

V_all,t = {l in {1, ..., 36} : the per-layer value at l is valid}
~~~

### Last layer

~~~text
cosine_gen_imgproto_hidden_last_layer_t
    = cosine_gen_imgproto_hidden_layer_36_t
~~~

### Early visual integration

~~~text
cosine_gen_imgproto_hidden_early_visual_integration_t
    = (1 / |V_early,t|) * sum_{l in V_early,t}
        cosine_gen_imgproto_hidden_layer_l_t

V_early,t = {l in {1, ..., 21} : the per-layer value at l is valid}
~~~

### Visual reasoning

~~~text
cosine_gen_imgproto_hidden_visual_reasoning_t
    = (1 / |V_reason,t|) * sum_{l in V_reason,t}
        cosine_gen_imgproto_hidden_layer_l_t

V_reason,t = {l in {22, ..., 34} : the per-layer value at l is valid}
~~~

---

## Edge cases and conventions

- If there are no image tokens, the image prototype is undefined and all four
  stored variants are `None`.
- If either `||x_t^l||_2 < 1e-8` or `||μ^l||_2 < 1e-8`, the per-layer cosine
  value is undefined and is recorded as `None` for that layer. An
  epsilon-protected cosine implementation must not turn a zero or near-zero
  vector into a normalized cosine value of `0.5`.
- Aggregated variants average over valid per-layer values only. If no valid
  values remain in an aggregate's layer range, that stored variant is `None`.
- The normalized values should be clipped to `[0, 1]` after normalization to
  protect against small floating-point excursions outside the cosine range.
