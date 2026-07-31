# VisConf

VisConf runs predictor-aligned visual-confidence experiments for Qwen2.5-VL-3B.

The normative design is documented in:

- `AGENTS.MD`
- `docs/REPO_SCHEMA.MD`
- `docs/OUTPUT_SCHEMA.md`
- The three metric documents in `docs/`

The initial experiment group contains six independent runs: one run for every
combination of MathVerse, MathVista, or MMMU-Pro with the diverse or concentrated
sampling strategy.

Phase 1 supports planning and validating this six-run group:

~~~bash
visconf plan --config configs/experiment_group.yaml
~~~

Model execution is introduced in later implementation phases.
