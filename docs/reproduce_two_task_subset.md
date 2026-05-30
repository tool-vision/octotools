# Reproduce the OpenAI-only Two-task Subset

This pipeline reproduces a budget-controlled subset of the OctoTools paper
workflow on `Game of 24` and `GPQA`. It is not the full 16-task paper table.

## Setup

Use Python 3.10+ in a clean environment.

```sh
conda create -n octotools python=3.10 -y
conda activate octotools
pip install -e . --no-deps
pip install -r requirements-repro.txt
```

Create `.env` in the repo root:

```sh
OPENAI_API_KEY=your_key_here
```

## Offline Check

```sh
python scripts/smoke_reproduce.py
python scripts/reproduce_octotools.py \
  --preset paper-two-task-500 \
  --dry-run \
  --label paper-two-task-dry-run
```

The dry run should report 40 jobs and an estimated 400 OpenAI requests with
`--max-steps 2`. The default request budget is 500 with a 100-request reserve,
so this preset is exactly at the planned daily execution limit.

## Real Run

```sh
python scripts/reproduce_octotools.py \
  --preset paper-two-task-500 \
  --engine gpt-4o-mini \
  --label paper-two-task-day1 \
  --max-steps 2 \
  --threads 1 \
  --request-budget 500
```

If network/API errors interrupt the run, resume without rerunning completed
outputs:

```sh
python scripts/reproduce_octotools.py \
  --preset paper-two-task-500 \
  --engine gpt-4o-mini \
  --label paper-two-task-day1 \
  --max-steps 2 \
  --threads 1 \
  --request-budget 500 \
  --skip-existing
```

## Compare With the Paper

```sh
python scripts/compare_paper_subset.py runs/paper-two-task-500/paper-two-task-day1
```

The comparison uses the paper Table 1 OctoTools values:

- Game of 24: `44.7 +/- 2.8`
- GPQA: `54.7 +/- 1.3`

The output files are:

- `paper_comparison.json`
- `paper_comparison.csv`

Subset accuracy is useful for reproducibility checks, but it is not
statistically comparable to the full paper result.
