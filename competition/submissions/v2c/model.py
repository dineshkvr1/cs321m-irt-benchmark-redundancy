"""NCF v2 submission for Predictive AI Evaluation Challenge.

Improvements over v1:
1. Retrained NCFHeadV2 with benchmark bias on clean HF data
2. Platt scaling calibration from K=5 adaptive labels (per PDF §3.4)
3. Graceful fallback for unseen benchmarks (no benchmark bias applied)
4. Wider clipping [0.001, 0.999] for better log-loss
5. Per-benchmark + per-subject fallback hierarchy
"""
from __future__ import annotations
import json, pickle, math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------- Module-level init (runs once) ----------
_DIR = Path(__file__).parent

# Load sentence transformer (declared in models.txt)
ENCODER = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
EMB_DIM = 768

# Load benchmark mapping
with open(_DIR / "benchmark_map.json") as f:
    BENCHMARK_LIST = json.load(f)
BENCHMARK_TO_IDX = {b: i for i, b in enumerate(BENCHMARK_LIST)}
N_BENCHMARKS = len(BENCHMARK_LIST)

# Load NCF head v2
class NCFHeadV2(nn.Module):
    def __init__(self, emb_dim, n_benchmarks):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
        )
        self.gmf_proj = nn.Linear(emb_dim, 128)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.benchmark_bias = nn.Embedding(n_benchmarks, 1)

    def forward(self, u, v, bench_idx=None):
        mlp_out = self.mlp(torch.cat([u, v], dim=-1))
        gmf_out = self.gmf_proj(u * v)
        combined = torch.cat([mlp_out, gmf_out], dim=-1)
        logit = self.head(combined).squeeze(-1)
        # Only apply benchmark bias if index is known (not None)
        if bench_idx is not None:
            logit = logit + self.benchmark_bias(bench_idx).squeeze(-1)
        return logit

NCF = NCFHeadV2(EMB_DIM, N_BENCHMARKS)
NCF.load_state_dict(torch.load(_DIR / "ncf_head.pt", map_location="cpu", weights_only=True))
NCF.eval()

# Load statistics for fallback
with open(_DIR / "stats.pkl", "rb") as f:
    _stats = pickle.load(f)
SUBJECT_MEANS = _stats["subject_means"]
BENCHMARK_MEANS = _stats["benchmark_means"]
BENCH_SUBJ_MEANS = _stats["benchmark_subject_means"]
GLOBAL_MEAN = _stats["global_mean"]

# Cache embeddings
_emb_cache = {}

def _encode(text: str) -> np.ndarray:
    if text not in _emb_cache:
        _emb_cache[text] = ENCODER.encode(text, convert_to_numpy=True)
    return _emb_cache[text]

# ---------- Platt scaling from labeled examples ----------
# Module-level cache: fitted once per round from the first predict() call
_platt_params = None  # (a, b) or None if not yet fitted / not enough data

def _fit_platt(labeled: list[dict]) -> tuple[float, float] | None:
    """Fit 1-parameter Platt scaling: calibrated_p = sigmoid(a * logit + b).
    Uses the K=5 labeled examples to fit a and b via simple grid search.
    Returns (a, b) or None if insufficient data."""
    if not labeled or len(labeled) < 2:
        return None

    # Get NCF raw logits for labeled examples
    logits = []
    labels = []
    for ex in labeled:
        try:
            u = _encode(ex["subject_content"])
            v = _encode(ex["item_content"])
            u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
            v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            benchmark = ex.get("benchmark", "")
            bench_idx = BENCHMARK_TO_IDX.get(benchmark)
            b_t = torch.tensor([bench_idx], dtype=torch.long) if bench_idx is not None else None
            with torch.no_grad():
                logit = NCF(u_t, v_t, b_t).item()
            logits.append(logit)
            labels.append(float(ex.get("label", 0.5)))
        except Exception:
            continue

    if len(logits) < 2:
        return None

    # Need both classes for meaningful calibration
    if len(set(labels)) < 2:
        return None

    logits = np.array(logits)
    labels = np.array(labels)

    # Simple Platt: find a, b that minimize log-loss of sigmoid(a*logit + b)
    # Use scipy if available, otherwise grid search
    best_loss = float("inf")
    best_a, best_b = 1.0, 0.0
    for a in np.linspace(0.5, 2.0, 16):
        for b in np.linspace(-1.0, 1.0, 21):
            p = 1.0 / (1.0 + np.exp(-(a * logits + b)))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            loss = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
            if loss < best_loss:
                best_loss = loss
                best_a, best_b = a, b

    return (best_a, best_b)


def _platt_calibrate(logit: float, labeled: list[dict] | None) -> float:
    """Apply Platt scaling if labeled examples are available."""
    global _platt_params

    if not labeled:
        return 1.0 / (1.0 + math.exp(-logit))

    # Fit once per round (cached at module level)
    if _platt_params is None:
        _platt_params = _fit_platt(labeled)

    if _platt_params is not None:
        a, b = _platt_params
        calibrated_logit = a * logit + b
        return 1.0 / (1.0 + math.exp(-calibrated_logit))
    else:
        return 1.0 / (1.0 + math.exp(-logit))


# ---------- Required entry point ----------
def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Pure subject-mean baseline — no NCF. Tests whether NCF helps or hurts."""
    benchmark = input.get("benchmark", "")
    subject_content = input.get("subject_content", "")

    # Use most specific prior available
    prob = BENCH_SUBJ_MEANS.get(
        (benchmark, subject_content),
        SUBJECT_MEANS.get(
            subject_content,
            BENCHMARK_MEANS.get(benchmark, GLOBAL_MEAN)
        )
    )

    # If labeled examples available, shift toward observed rate
    if labeled and len(labeled) >= 2:
        labels = [float(ex.get("label", 0.5)) for ex in labeled]
        obs_mean = np.mean(labels)
        prob = 0.6 * prob + 0.4 * obs_mean

    return float(max(0.10, min(0.90, prob)))
