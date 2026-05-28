"""Uncertainty-based adaptive labeling.

Prioritize (subject, item) pairs where the model is most uncertain
(predicted probability closest to 0.5). This maximizes the information
gained from the K=5 revealed labels per data category.
"""
from __future__ import annotations

# Import the prediction model to compute uncertainty
try:
    from model import _encode, NCF, BENCHMARK_TO_IDX, SUBJECT_MEANS, BENCHMARK_MEANS, GLOBAL_MEAN
    import torch
    _MODEL_AVAILABLE = True
except Exception:
    _MODEL_AVAILABLE = False


def acquisition_function(input: dict) -> float:
    """Return a labeling-priority score. Higher = more desired for labeling.
    
    We score by uncertainty: pairs closest to P=0.5 get highest scores.
    This is equivalent to maximum entropy sampling.
    """
    if not _MODEL_AVAILABLE:
        import random
        return random.random()

    try:
        subject_content = input.get("subject_content", "")
        item_content = input.get("item_content", "")
        benchmark = input.get("benchmark", "")

        u = _encode(subject_content)
        v = _encode(item_content)
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
        bench_idx = BENCHMARK_TO_IDX.get(benchmark)
        b_t = torch.tensor([bench_idx], dtype=torch.long) if bench_idx is not None else None

        with torch.no_grad():
            logit = NCF(u_t, v_t, b_t)
            prob = torch.sigmoid(logit).item()

        # Uncertainty = 1 - |prob - 0.5| * 2  →  highest at P=0.5
        uncertainty = 1.0 - abs(prob - 0.5) * 2.0
        return float(uncertainty)

    except Exception:
        return 0.5  # moderate priority on error
