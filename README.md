# CS321M Course Project: IRT-Based Redundancy Analysis of LLM Benchmarks

**Course:** CS321M – Measurement Science (Stanford, Spring 2026)  
**Author:** Dinesh Katupputhur Ramprasath (din1993@stanford.edu)

## Overview

This project applies Item Response Theory (IRT) to analyze redundancy in six LLM benchmarks from the Fantastic-Bugs dataset. Key finding: **IRT-guided item selection recovers model rankings using as few as 5% of benchmark items** (Spearman ρ > 0.95) for well-designed benchmarks.

The project also includes the Predictive AI Evaluation Challenge submission (content-based NCF), achieving a score of **−0.61** (negative log-loss) vs. baseline −0.79.

## Repository Structure

```
├── notebooks/
│   ├── 01_fantastic_bugs_irt_analysis.ipynb   # Main IRT analysis (Rasch, 2PL, LogisticFM)
│   └── 02_competition_submission.ipynb        # Predictive AI Evaluation competition
├── competition/
│   ├── model.py              # Best submission entry point (v2)
│   ├── labeling.py           # Uncertainty-based acquisition function
│   ├── train_ncf_v2.py       # Training script for NCF v2
│   └── submissions/          # All ablation variants
│       ├── v1/               # Original (λ=1.0, clip [0.001,0.999]) → -1.32
│       ├── v2_best/          # Best (λ=0.5, clip [0.05,0.95])     → -0.61
│       ├── v2a/              # Ablation (λ=0.6, clip [0.03,0.97]) → -0.62
│       ├── v2b/              # Ablation (λ=0.4, clip [0.10,0.90]) → -0.62
│       └── v2c/              # No NCF (λ=0.0, clip [0.10,0.90])   → -0.64
├── figures/                   # All generated figures (PDF)
├── requirements.txt
└── README.md
```

## Reproducing Results

### Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### IRT Analysis (Notebook 01)

1. Open `notebooks/01_fantastic_bugs_irt_analysis.ipynb`
2. The notebook downloads the Fantastic-Bugs dataset from HuggingFace automatically
3. Run all cells sequentially — generates all figures in `figures/`
4. **Runtime:** ~15 minutes on a single GPU (A100); CPU is also supported (~45 min)
5. **Random seed:** 42 (set in notebook)

### Competition Model Training

```bash
python competition/train_ncf_v2.py
```

- Downloads `aims-foundations/measurement-db` from HuggingFace
- Trains NCF v2 for 30 epochs (~40 min on A100 including embedding precomputation)
- Outputs: `ncf_head.pt` (trained weights), `stats.pkl` (precomputed means)

### Competition Submission

Each variant in `competition/submissions/` contains `model.py` and `labeling.py` that match the Codabench submission format. See the competition report for ablation results.

## Key Results

### IRT Benchmark Redundancy Analysis

| Benchmark | Items for ρ > 0.95 | Fraction | Assessment |
|-----------|-------------------|----------|------------|
| MMLU | 28 | 5% | Highly reducible |
| MedQA | 49 | 5% | Highly reducible |
| OpenBookQA | 50 | 10% | Moderately reducible |
| LegalBench | 199 | 10% | Moderately reducible |
| BoolQ | 1,658 | 50% | Poorly reducible |
| BBQ | — | >100% | Irreducible |

- **Discrimination concentration** (Gini coefficient) is the strongest predictor of reducibility (r = −0.94)
- Five item selection strategies compared: random, stratified, max-info (tinyBenchmarks), integrated info, difficulty-coverage
- IRT-guided selection outperforms random by 0.18 in Spearman ρ at 5% items (MMLU)

### Predictive AI Evaluation Competition

| Variant | NCF weight (λ) | Clip | Neg Log-Loss |
|---------|-----------------|------|-------------|
| v2 (best) | 0.50 | [0.05, 0.95] | **−0.61** |
| v2a | 0.60 | [0.03, 0.97] | −0.62 |
| v2b | 0.40 | [0.10, 0.90] | −0.62 |
| v2c (no NCF) | 0.00 | [0.10, 0.90] | −0.64 |
| v1 (original) | 1.00 | [0.001, 0.999] | −1.32 |
| Baseline | — | — | −0.79 |

## Data Sources

- **IRT Analysis:** [Fantastic-Bugs dataset](https://huggingface.co/datasets/stair-lab/fantastic-bugs) (6 benchmarks, 42–91 models, 500–3,316 items)
- **Competition:** [Measurement database](https://huggingface.co/datasets/aims-foundations/measurement-db) (5.36M responses, 909 subjects, 103K items)

## Dependencies

- Python 3.8+, PyTorch 2.1, NumPy 1.24, SciPy 1.10, Matplotlib 3.7
- [torch_measure](https://github.com/aims-foundations/torch_measure) (IRT model fitting)
- sentence-transformers 2.2.2 (competition model embeddings)
- See `requirements.txt` for exact versions

## Compute Environment

Single Azure ML compute instance (Standard_NC24ads_A100_v4, NVIDIA A100 40GB GPU).  
Full IRT pipeline: ~15 min. Competition training: ~40 min.
