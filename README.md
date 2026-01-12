# Mixed-Initiative LookOut (CHI’99-inspired)

This repository contains a **self-contained Python implementation** of a **synthetic mixed-initiative decision-making model** inspired by the ideas in Eric Horvitz’s CHI’99 *LookOut* work on attention, interruption, and decision-theoretic control.

The code generates **synthetic interaction traces**, analytic decision thresholds, figures, and dataset exports suitable for research in:
- mixed-initiative systems,
- decision theory under uncertainty,
- attention-aware assistants,
- calibration and interpretability,
- reinforcement learning and policy analysis.

**Developed by DBbun LLC — January 2026.**

> This project is inspired by publicly described concepts in the literature.  
> It is **not affiliated with Microsoft Research or Eric Horvitz** and does **not** include any original LookOut code or data.

---

## What this repository contains

- **`chi99horvitz v1.0.py`**  
  A single, runnable Python script that:
  - generates synthetic mixed-initiative episodes,
  - computes expected utilities and analytic thresholds,
  - simulates attention and interruption costs,
  - produces figures (policy maps, threshold curves, attention dynamics),
  - exports datasets in CSV and JSONL formats,
  - writes a quality / validation report.

There are **no subdirectories** and no external assets required.

---

## Conceptual overview

The model studies how an intelligent assistant should choose between:
- **no action**,
- **dialog / clarification**,
- **scoped suggestions**,
- **direct action**,

based on:
- uncertainty about user intent,
- user attention and readiness,
- urgency / time pressure,
- asymmetric utilities and costs.

Each decision is made by **explicit expected-utility maximization**, and the resulting **decision thresholds are stored analytically**, enabling interpretability and counterfactual analysis.

---

## Outputs produced by the script

When run, the script produces:

- **Synthetic datasets**
  - `mixed_initiative_traces.csv` — tabular dataset
  - `mixed_initiative_traces.jsonl` — structured episode-level format

- **Figures**
  - Expected-utility threshold plots
  - Context-shifted policy curves
  - Attention (dwell time vs message length) dynamics
  - Mixed-initiative policy maps

- **Quality report**
  - Calibration metrics (AUROC, ECE)
  - Sensitivity and monotonicity checks
  - Baseline policy comparisons

- **Configuration**
  - `mixed_initiative_traces.config.json` capturing all generator parameters

---

## How to run

```bash
python "chi99horvitz v1.0.py"
