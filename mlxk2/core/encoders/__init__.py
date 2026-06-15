"""Vendored encoder-model code (ADR-015 Slice B).

`mlx-lm` loads only causal/decoder models, so encoder embedders (BERT/XLM-R family:
bge, e5, MiniLM, ...) need their own forward pass. This subpackage holds the first — and
deliberately isolated — model-architecture code mlx-knife carries: a bounded, conscious
exception to ADR-023 (the verified list normally never exceeds upstream), justified in
ADR-015 §Decision: Implementation Library. Attribution lives in ``mlxk2/NOTICE``.
"""
