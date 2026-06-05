"""NCF submission for Predictive AI Evaluation Challenge."""
from __future__ import annotations
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------- Module-level init (runs once) ----------
_DIR = Path(__file__).parent

# Load sentence transformer (declared in models.txt, pre-fetched by platform)
ENCODER = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
EMB_DIM = 768

# Load trained NCF head
class NCFHead(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
    def forward(self, u, v):
        x = torch.cat([u, v], dim=-1)
        return self.net(x).squeeze(-1)

NCF = NCFHead(EMB_DIM)
NCF.load_state_dict(torch.load(_DIR / "ncf_head.pt", map_location="cpu", weights_only=True))
NCF.eval()

# Load per-subject mean fallback
with open(_DIR / "subject_means.pkl", "rb") as f:
    _sm = pickle.load(f)
SUBJECT_MEANS = _sm["means"]
GLOBAL_MEAN = _sm["global"]

# Cache embeddings to avoid re-encoding the same text
_emb_cache = {}

def _encode(text: str) -> np.ndarray:
    if text not in _emb_cache:
        _emb_cache[text] = ENCODER.encode(text, convert_to_numpy=True)
    return _emb_cache[text]

# ---------- Required entry point ----------
def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return predicted P(subject answers item correctly)."""
    try:
        u = _encode(input["subject_content"])
        v = _encode(input["item_content"])
        u_t = torch.tensor(u, dtype=torch.float32).unsqueeze(0)
        v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logit = NCF(u_t, v_t)
            prob = torch.sigmoid(logit).item()
        # Clip to avoid log(0)
        return float(max(0.01, min(0.99, prob)))
    except Exception:
        # Fallback to per-subject mean
        mean = SUBJECT_MEANS.get(input.get("subject_content", ""), GLOBAL_MEAN)
        return float(max(0.01, min(0.99, mean)))
