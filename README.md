# CS321M Course Project: IRT-Based Redundancy Analysis of LLM Benchmarks

**Course:** CS321M – Measurement Science (Stanford, Spring 2026)  
**Author:** Dinesh Katupputhur Ramprasath (din1993@stanford.edu)

## Overview

This project applies Item Response Theory (IRT) to analyze redundancy in six LLM benchmarks from the Fantastic-Bugs dataset. Key finding: **IRT-guided item selection recovers model rankings using as few as 5% of benchmark items** (Spearman ρ > 0.95) for well-designed benchmarks.

## Repository Structure

```
├── notebooks/
│   ├── 01_fantastic_bugs_irt_analysis.ipynb   # Main IRT analysis (Rasch, 2PL, LogisticFM)
│   └── 02_competition_submission.ipynb        # Predictive AI Evaluation competition
├── competition/
│   ├── model.py          # NCF v2 model (best submission, score: -0.61)
│   ├── labeling.py       # Uncertainty-based acquisition for adaptive labeling
│   └── train_ncf_v2.py   # Training script for NCF v2
├── figures/               # All generated figures (PDF)
│   ├── difficulty_distributions.pdf
│   ├── discrimination_distributions.pdf
│   ├── test_information.pdf
│   ├── tiny_benchmarks_recovery.pdf
│   ├── theta_correlations.pdf
│   └── ...
└── README.md
```

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

- Discrimination concentration (Gini coefficient) is the strongest predictor of reducibility (r = −0.94)
- Five item selection strategies compared: random, stratified, max-info (tinyBenchmarks), integrated info, difficulty-coverage

### Predictive AI Evaluation Competition

- **Model:** Neural Collaborative Filtering (NCF) v2 with sentence-transformer embeddings
- **Best Score:** −0.61 (negative log-loss), baseline: −0.79, #1: −0.56
- **Key features:** Shrinkage toward subject mean (λ=0.5), benchmark bias, hierarchical fallback

## Data

- **IRT Analysis:** [Fantastic-Bugs dataset](https://huggingface.co/datasets/stair-lab/fantastic-bugs) (6 benchmarks, 42–91 models, 500–3,316 items)
- **Competition:** [Measurement database](https://huggingface.co/datasets/aims-foundations/measurement-db) (5.36M responses, 909 subjects, 103K items)

## Dependencies

- Python 3.8+, PyTorch, NumPy, SciPy, Matplotlib
- [torch_measure](https://github.com/aims-foundations/torch_measure) (IRT fitting)
- sentence-transformers (competition model)
