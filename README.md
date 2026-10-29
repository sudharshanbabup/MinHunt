# MinHint: Hint Ladder Planning Under a Turn Budget

This repository contains the Python implementation and experimental evaluation for the paper:
**"MinHint: Hint Ladder Planning Under a Turn Budget for Socratic Tutoring Agents"**.

MinHint formalizes hint escalation on an ordered ladder of $K$ assistance levels as a shortest-path problem under a dialogue turn budget, solved via Lagrangian relaxation.

## Requirements

Python 3.9+ with standard scientific libraries:
```bash
pip install numpy pandas scipy scikit-learn torch
```

## Data Download

Download the two benchmark tutoring logs (ASSISTments 2009–2010 and Cognitive Tutor):
```bash
python download_data.py
```
This downloads `ct.csv` and `as.csv` into a local `data/` directory.

## Audits & Verification

Run the mathematical verification suite (shortest-path DP vs. brute force, closed forms, odds rule, Lagrangian duality, cost bounds, re-planning, and value of context):
```bash
python test_ladder.py
```

Run the dialogue simulation and evaluation audit:
```bash
python audit_eval.py
```

## Running Experiments

Run the policy evaluation experiments across seeds:
```bash
# Cognitive Tutor benchmark (10 seeds)
python experiment.py ct --seeds 10

# ASSISTments benchmark (run seeds in two batches or in parallel)
python experiment.py assist --seed_from 0 --seeds 5
python experiment.py assist --seed_from 5 --seeds 10

# Merge parallel runs
python merge.py assist ct

# View summary table and metrics
python summarize.py ct assist
```

## Code Overview

- `ladder.py`: Core MinHint planner, cost matrices, shortest-path DP ($O(K^2)$), and Lagrangian dual calibration.
- `data.py`: Preprocessing, causal history feature extraction, and learner-level splits.
- `download_data.py`: Downloads the public tutoring benchmark datasets.
- `test_ladder.py`: Mathematical audits and theoretical assertions.
- `audit_eval.py`: Turn-by-turn dialogue simulation audit vs. vectorized evaluation.
- `experiment.py`: Trains assistance-need neural network and evaluates policies across turn budgets.
- `merge.py`: Merges multi-seed experiment logs.
- `summarize.py`: Summarizes evaluation metrics and baselines across seeds.
- `export_figs.py`: Generates summary tables, latex metrics, and figure data.
