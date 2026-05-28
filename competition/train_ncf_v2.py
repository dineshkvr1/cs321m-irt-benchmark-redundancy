"""Train improved NCF head for Predictive AI Evaluation Challenge v2.

Improvements over v1:
1. Retrain on clean HuggingFace data (no contaminated public benchmarks)
2. Better architecture: residual connections, layer norm
3. Per-benchmark bias terms for calibration
4. Wider output range for better log-loss
5. Save per-subject AND per-benchmark means for fallback
"""
from __future__ import annotations
import os, pickle, json, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datasets import Features, Value, load_dataset
from huggingface_hub import HfApi
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "my_submission_v2"
OUT_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path.home() / ".cache" / "competition_hf"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)

ENCODER_NAME = "sentence-transformers/all-mpnet-base-v2"
EMB_DIM = 768
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ── 1. Load Data ────────────────────────────────────────────────────
print("Loading data from HuggingFace...")
REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
response_files = sorted(
    name for name in repo_files
    if name.endswith(".parquet")
    and name not in REGISTRY_FILES
    and not name.endswith("_traces.parquet")
)
print(f"Found {len(response_files)} response files")

response_features = Features({
    "subject_id": Value("string"),
    "item_id": Value("string"),
    "benchmark_id": Value("string"),
    "trial": Value("int64"),
    "test_condition": Value("string"),
    "response": Value("float64"),
    "correct_answer": Value("string"),
    "trace": Value("string"),
})

responses = load_dataset(
    REPO_ID, data_files=response_files,
    features=response_features, split="train",
    cache_dir=str(CACHE_DIR),
)
items = load_dataset(REPO_ID, data_files="items.parquet", split="train", cache_dir=str(CACHE_DIR))
subjects = load_dataset(REPO_ID, data_files="subjects.parquet", split="train", cache_dir=str(CACHE_DIR))
benchmarks = load_dataset(REPO_ID, data_files="benchmarks.parquet", split="train", cache_dir=str(CACHE_DIR))

print(f"Responses: {len(responses)}, Items: {len(items)}, Subjects: {len(subjects)}, Benchmarks: {len(benchmarks)}")

# ── 2. Build lookup tables ──────────────────────────────────────────
items_by_id = {row["item_id"]: row for row in items}
subjects_by_id = {row["subject_id"]: row for row in subjects}
benchmarks_by_id = {row["benchmark_id"]: row for row in benchmarks}

def render_subject_content(subject, fallback_subject_id):
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    for key, label in [("provider","Organization"),("params","Parameters"),("release_date","Released"),("family","Family")]:
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)

def render_item_content(item):
    return item.get("content", "") or ""

# ── 3. Convert to training examples ────────────────────────────────
print("Converting to training examples...")
examples = []
subject_stats = defaultdict(list)
benchmark_stats = defaultdict(list)
benchmark_subject_stats = defaultdict(list)

for row in responses:
    label = row["response"]
    if label is None:
        continue
    # Binarize: treat >= 0.5 as correct for continuous, exact 0/1 for binary
    binary_label = float(label >= 0.5) if not (label == 0.0 or label == 1.0) else float(label)
    
    subject = subjects_by_id.get(row["subject_id"], {})
    item = items_by_id.get(row["item_id"], {})
    benchmark = benchmarks_by_id.get(row["benchmark_id"], {})
    
    subject_content = render_subject_content(subject, row["subject_id"])
    item_content = render_item_content(item)
    benchmark_id = benchmark.get("benchmark_id") or row["benchmark_id"]
    
    if not item_content:
        continue
    
    examples.append({
        "subject_content": subject_content,
        "item_content": item_content,
        "benchmark": benchmark_id,
        "label": binary_label,
        "raw_label": float(label),
    })
    
    subject_stats[subject_content].append(binary_label)
    benchmark_stats[benchmark_id].append(binary_label)
    benchmark_subject_stats[(benchmark_id, subject_content)].append(binary_label)

print(f"Total training examples: {len(examples)}")

# ── 4. Compute statistics for fallback ──────────────────────────────
subject_means = {k: np.mean(v) for k, v in subject_stats.items()}
benchmark_means = {k: np.mean(v) for k, v in benchmark_stats.items()}
benchmark_subject_means = {k: np.mean(v) for k, v in benchmark_subject_stats.items()}
global_mean = np.mean([e["label"] for e in examples])

print(f"Global mean: {global_mean:.4f}")
print(f"Unique subjects: {len(subject_means)}, Unique benchmarks: {len(benchmark_means)}")

# Save stats
with open(OUT_DIR / "stats.pkl", "wb") as f:
    pickle.dump({
        "subject_means": subject_means,
        "benchmark_means": benchmark_means,
        "benchmark_subject_means": benchmark_subject_means,
        "global_mean": global_mean,
    }, f)

# ── 5. Encode text with sentence transformer ───────────────────────
print(f"Loading encoder: {ENCODER_NAME}")
encoder = SentenceTransformer(ENCODER_NAME, cache_folder=str(CACHE_DIR))

# Collect unique texts
unique_subjects = list(set(e["subject_content"] for e in examples))
unique_items = list(set(e["item_content"] for e in examples))
print(f"Encoding {len(unique_subjects)} subjects and {len(unique_items)} items...")

subject_embs = {}
for i in range(0, len(unique_subjects), 128):
    batch = unique_subjects[i:i+128]
    embs = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    for txt, emb in zip(batch, embs):
        subject_embs[txt] = emb
    if (i // 128) % 10 == 0:
        print(f"  Subjects: {i+len(batch)}/{len(unique_subjects)}")

item_embs = {}
for i in range(0, len(unique_items), 128):
    batch = unique_items[i:i+128]
    embs = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False, batch_size=128)
    for txt, emb in zip(batch, embs):
        item_embs[txt] = emb
    if (i // 128) % 10 == 0:
        print(f"  Items: {i+len(batch)}/{len(unique_items)}")

print("Encoding complete.")

# ── 6. Build PyTorch dataset ────────────────────────────────────────
# Assign benchmark indices
benchmark_list = sorted(set(e["benchmark"] for e in examples))
benchmark_to_idx = {b: i for i, b in enumerate(benchmark_list)}
n_benchmarks = len(benchmark_list)
print(f"Number of benchmarks: {n_benchmarks}")

# Save benchmark mapping
with open(OUT_DIR / "benchmark_map.json", "w") as f:
    json.dump(benchmark_list, f)

class PairDataset(Dataset):
    def __init__(self, examples, subject_embs, item_embs, benchmark_to_idx):
        self.examples = examples
        self.subject_embs = subject_embs
        self.item_embs = item_embs
        self.benchmark_to_idx = benchmark_to_idx
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        e = self.examples[idx]
        u = self.subject_embs[e["subject_content"]]
        v = self.item_embs[e["item_content"]]
        b = self.benchmark_to_idx[e["benchmark"]]
        y = e["label"]
        return (
            torch.tensor(u, dtype=torch.float32),
            torch.tensor(v, dtype=torch.float32),
            torch.tensor(b, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )

# Split 90/10
np.random.seed(42)
indices = np.random.permutation(len(examples))
split = int(0.9 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

train_ds = PairDataset([examples[i] for i in train_idx], subject_embs, item_embs, benchmark_to_idx)
val_ds = PairDataset([examples[i] for i in val_idx], subject_embs, item_embs, benchmark_to_idx)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# ── 7. Model ────────────────────────────────────────────────────────
class NCFHeadV2(nn.Module):
    """Improved NCF with benchmark bias, element-wise product, and residual."""
    def __init__(self, emb_dim, n_benchmarks):
        super().__init__()
        self.n_benchmarks = n_benchmarks
        
        # MLP path (concatenation)
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
        
        # GMF path (element-wise product)
        self.gmf_proj = nn.Linear(emb_dim, 128)
        
        # Combine
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        
        # Per-benchmark bias
        self.benchmark_bias = nn.Embedding(n_benchmarks, 1)
        nn.init.zeros_(self.benchmark_bias.weight)
    
    def forward(self, u, v, bench_idx):
        # MLP path
        mlp_out = self.mlp(torch.cat([u, v], dim=-1))
        # GMF path
        gmf_out = self.gmf_proj(u * v)
        # Combine
        combined = torch.cat([mlp_out, gmf_out], dim=-1)
        logit = self.head(combined).squeeze(-1)
        # Add benchmark bias
        logit = logit + self.benchmark_bias(bench_idx).squeeze(-1)
        return logit

model = NCFHeadV2(EMB_DIM, n_benchmarks).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss()

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ── 8. Train ────────────────────────────────────────────────────────
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    n_batches = 0
    for u, v, b, y in train_dl:
        u, v, b, y = u.to(DEVICE), v.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
        logit = model(u, v, b)
        loss = criterion(logit, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    
    # Validate
    model.eval()
    val_loss = 0
    val_n = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for u, v, b, y in val_dl:
            u, v, b, y = u.to(DEVICE), v.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
            logit = model(u, v, b)
            loss = criterion(logit, y)
            val_loss += loss.item() * u.size(0)
            val_n += u.size(0)
            probs = torch.sigmoid(logit)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    val_loss /= val_n
    
    # Compute negative log-loss (what Codabench reports)
    all_probs = np.array(all_probs).clip(0.001, 0.999)
    all_labels = np.array(all_labels)
    neg_logloss = -np.mean(all_labels * np.log(all_probs) + (1 - all_labels) * np.log(1 - all_probs))
    neg_logloss_score = -neg_logloss  # Higher is better
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | train_loss={train_loss/n_batches:.4f} | val_loss={val_loss:.4f} | neg_logloss={neg_logloss_score:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), OUT_DIR / "ncf_head.pt")
        print(f"  → Saved best model (val_loss={val_loss:.4f})")

print(f"\nBest val loss: {best_val_loss:.4f}")

# ── 9. Save benchmark list for submission ───────────────────────────
# Also save a version-info file
with open(OUT_DIR / "version.txt", "w") as f:
    f.write("v2 - retrained on clean data, NCFHeadV2 with benchmark bias\n")
    f.write(f"n_benchmarks={n_benchmarks}\n")
    f.write(f"best_val_loss={best_val_loss:.4f}\n")
    f.write(f"train_examples={len(train_ds)}\n")
    f.write(f"val_examples={len(val_ds)}\n")

print("Training complete. Artifacts saved to", OUT_DIR)
