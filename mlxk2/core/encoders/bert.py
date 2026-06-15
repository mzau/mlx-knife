"""Minimal MIT BERT encoder for embeddings (ADR-015 Slice B).

Vendored, self-contained ``mlx.nn`` re-implementation of the BERT encoder body, adapted from
Apple's MIT-licensed ``mlx-examples`` BERT (attribution in ``mlxk2/NOTICE``). It exists because
``mlx-lm`` loads only causal/decoder models — encoder embedders (bge, e5, MiniLM = ``model_type:
bert``) have no upstream MLX loader. mlx-knife owns the code so it does not break on the next
``mlx-lm`` bump.

**Key-name contract (load-bearing):** mlx-community BERT conversions keep *native HuggingFace*
state-dict keys (verified against bge-small-en-v1.5-4bit and multilingual-e5-small-mlx):
``embeddings.word_embeddings.weight``, ``encoder.layer.N.attention.self.query.{weight,bias}``,
``pooler.dense.*``, ... Every module attribute below is named so the flattened parameter tree
equals those keys, and weights load with **zero remap**. Renaming any attribute (e.g. ``self`` →
``attn``, or ``layer`` → ``layers``) silently breaks the strict load — do not.

Scope (Slice B): single sequence at a time (batch 1), **no attention mask** (all tokens real, no
padding), absolute positions ``0..L-1`` (``model_type: bert``; the RoBERTa +2 padding-idx offset
does NOT apply here — xlm-roberta is deliberately out of scope). Pooling/normalization happen in
the caller (``EmbeddingRunner``); this module returns the last hidden state.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class _BertConfig:
    """Plain holder for the BERT dims read from ``config.json`` (HF field names)."""

    def __init__(self, c: Dict[str, Any]):
        self.vocab_size = int(c["vocab_size"])
        self.hidden_size = int(c["hidden_size"])
        self.num_hidden_layers = int(c["num_hidden_layers"])
        self.num_attention_heads = int(c["num_attention_heads"])
        self.intermediate_size = int(c["intermediate_size"])
        self.max_position_embeddings = int(c.get("max_position_embeddings", 512))
        self.type_vocab_size = int(c.get("type_vocab_size", 2))
        self.layer_norm_eps = float(c.get("layer_norm_eps", 1e-12))
        self.hidden_act = str(c.get("hidden_act", "gelu"))


def _act_fn(name: str):
    """Map a HF ``hidden_act`` to the exact MLX activation. BERT's ``gelu`` is the erf form —
    the tanh approximation silently shifts every activation and degrades retrieval quality."""
    a = (name or "gelu").lower()
    if a in ("gelu_new", "gelu_pytorch_tanh", "gelu_fast", "gelu_approximate"):
        return nn.gelu_approx
    if a == "relu":
        return nn.relu
    return nn.gelu  # exact (erf) — default for model_type: bert


class BertSelfAttention(nn.Module):
    def __init__(self, cfg: _BertConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def __call__(self, x):
        b, seqlen, h = x.shape
        nh = self.num_heads
        d = h // nh
        scale = 1.0 / math.sqrt(d)
        # (b, seqlen, h) -> (b, nh, seqlen, d)
        q = self.query(x).reshape(b, seqlen, nh, d).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(b, seqlen, nh, d).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(b, seqlen, nh, d).transpose(0, 2, 1, 3)
        # Full (unmasked) self-attention; single sequence, no padding.
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        weights = mx.softmax(scores, axis=-1)
        ctx = weights @ v  # (b, nh, seqlen, d)
        return ctx.transpose(0, 2, 1, 3).reshape(b, seqlen, h)


class BertSelfOutput(nn.Module):
    def __init__(self, cfg: _BertConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, hidden, input_tensor):
        return self.LayerNorm(self.dense(hidden) + input_tensor)


class BertAttention(nn.Module):
    def __init__(self, cfg: _BertConfig):
        super().__init__()
        # Attribute name MUST be "self" to match the HF key ``attention.self.*`` — direct
        # attribute assignment registers the child under that name. Do not rename.
        self.self = BertSelfAttention(cfg)
        self.output = BertSelfOutput(cfg)

    def __call__(self, x):
        return self.output(self.self(x), x)


class BertIntermediate(nn.Module):
    def __init__(self, cfg: _BertConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self._act = _act_fn(cfg.hidden_act)

    def __call__(self, x):
        return self._act(self.dense(x))


class BertOutput(nn.Module):
    def __init__(self, cfg: _BertConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, hidden, input_tensor):
        return self.LayerNorm(self.dense(hidden) + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, cfg: _BertConfig):
        super().__init__()
        self.attention = BertAttention(cfg)
        self.intermediate = BertIntermediate(cfg)
        self.output = BertOutput(cfg)

    def __call__(self, x):
        a = self.attention(x)
        return self.output(self.intermediate(a), a)


class BertEncoder(nn.Module):
    def __init__(self, cfg: _BertConfig):
        super().__init__()
        # List attribute named "layer" -> flattened keys encoder.layer.0 ... encoder.layer.N-1.
        self.layer = [BertLayer(cfg) for _ in range(cfg.num_hidden_layers)]

    def __call__(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class BertEmbeddings(nn.Module):
    def __init__(self, cfg: _BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.token_type_embeddings = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, input_ids):
        length = input_ids.shape[1]
        positions = mx.arange(length)[None]                       # (1, L), 0..L-1 (absolute)
        token_type = mx.zeros((1, length), dtype=input_ids.dtype)  # single sequence -> all zeros
        e = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)
            + self.token_type_embeddings(token_type)
        )
        return self.LayerNorm(e)


class BertPooler(nn.Module):
    """The HF tanh pooler head. Loaded for a clean strict-load, but **never called** — sentence
    embeddings use the CLS token of the last hidden state or a mean pool, not this head."""

    def __init__(self, cfg: _BertConfig):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def __call__(self, hidden):
        return mx.tanh(self.dense(hidden[:, 0]))


class BertModel(nn.Module):
    def __init__(self, cfg: _BertConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)
        self.pooler = BertPooler(cfg)

    def __call__(self, input_ids):
        """Return the last hidden state ``(1, L, hidden)`` (pre-pooling)."""
        return self.encoder(self.embeddings(input_ids))


# HF registers these as runtime buffers, not learnable params; they are recomputed in the
# forward (positions = arange, token_type = zeros), so drop them before a strict load. The
# quantized bge export omits them entirely; the float e5 export ships ``embeddings.position_ids``.
_NON_PARAM_BUFFER_LEAVES = ("position_ids", "token_type_ids")


def build_bert(config: Dict[str, Any]) -> BertModel:
    """Build the float BERT module tree from a HF ``config.json`` dict (no weights)."""
    return BertModel(_BertConfig(config))


def _load_safetensors(model_dir: Path) -> Dict[str, Any]:
    """Load + merge all weight shards as MLX arrays (honors an index, else globs ``*.safetensors``)."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted({model_dir / f for f in weight_map.values()})
    else:
        files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors weights found in {model_dir}")
    weights: Dict[str, Any] = {}
    for f in files:
        weights.update(mx.load(str(f)))
    return weights


def load_bert(model_dir) -> Tuple[BertModel, Dict[str, Any]]:
    """Load a BERT encoder from a resolved model directory.

    Mirrors the ``mlx-lm`` load idiom: build float → (optionally) quantize from the config →
    strict-load HF-named weights → realize. Returns ``(model, config)``.
    """
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    model = build_bert(config)

    weights = _load_safetensors(model_dir)
    weights = {
        k: v for k, v in weights.items()
        if k.rsplit(".", 1)[-1] not in _NON_PARAM_BUFFER_LEAVES
    }

    # Quantize BEFORE loading so module shapes (packed uint32 + scales/biases) match the on-disk
    # tensors. The predicate quantizes exactly the modules whose scales are present — Linear and
    # Embedding that were exported quantized — and leaves LayerNorm float. Gated on the config
    # block, so a float model (e5) skips quantization entirely.
    q = config.get("quantization")
    if q:
        nn.quantize(
            model,
            group_size=int(q.get("group_size", 64)),
            bits=int(q.get("bits", 4)),
            class_predicate=lambda path, module: (
                hasattr(module, "to_quantized") and f"{path}.scales" in weights
            ),
        )

    # Loud, specific failure if the module tree and the on-disk keys disagree (the real R1 guard —
    # unit tests can't load real weights, so this is the first place a key-name bug surfaces).
    model_keys = {k for k, _ in tree_flatten(model.parameters())}
    provided = set(weights)
    missing, extra = model_keys - provided, provided - model_keys
    if missing or extra:
        raise ValueError(
            f"BERT weight-key mismatch for '{model_dir.name}': "
            f"missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}"
        )

    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model, config
