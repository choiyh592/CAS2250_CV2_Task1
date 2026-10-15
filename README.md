# Diffusion toy project: 2D spiral → MNIST

CAS2250 Introduction to AI, task 2. DDPM-style diffusion trained from scratch on a 2D spiral (Tier 0) and on class-conditional MNIST with classifier-free guidance (Tier 1), sampled with DDIM.

Every number in the report comes from the commands in this README. All runs used seeds **0, 1 and 2**, and reported values are mean ± standard deviation over those three seeds.

## What we changed from the starter code

| File | Change |
|---|---|
| `train.py` | Filled in `training_step` (TODO 1). Added `--p_uncond` (default 0.1) and `--base` (default 96) options for the ablations; defaults keep the starter behaviour. Checkpoints and `*_train.json` also record `lr`, `p_uncond`, `base`, `epochs`. |
| `sample.py` | Filled in `sample_loop` (TODO 2). `load()` reads the model width from the checkpoint (`base`, default 96). |
| `complete_assignment.py` | New. Runs every command below in order, then collects `results/` and writes `results/summary.md`. |
| `report_figures.py` | New. Report figures and the two sampling-only experiments. |
| `data.py`, `diffusion.py`, `models.py`, `metrics.py`, `eval.py` | Unchanged. |

The course's original README is kept as `STARTER_README.md`.

## Environment

Python 3.11.15, PyTorch 2.14.1 with CUDA 12.6, one NVIDIA RTX 3050 Laptop GPU (4 GB), Windows. We managed the environment with [uv](https://docs.astral.sh/uv/).

```bash
uv venv .venv --python 3.11
uv pip install --python .venv torch==2.14.1 torchvision==0.29.1 --torch-backend=cu126
uv pip install --python .venv -r requirements.txt
```

With plain pip, `pip install -r requirements.txt` also works. On Windows it installs a CPU-only torch from PyPI; for CUDA, first run `pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126`. MNIST downloads itself into `data/` on first use.

Activate the environment before the commands below (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere).

## Reproduce everything with one command

```bash
python complete_assignment.py
```

This runs every command in the next section, skipping any step whose output already exists, so it can be restarted after an interruption. It then copies the results into `results/`, writes `results/summary.md`, and runs `report_figures.py`. Each step's output goes to `logs/<step>.txt`. On our GPU the whole run took about 12 hours: the main Tier 1 runs take about 32 minutes each, width 128 about 44 minutes, and each Tier 1 evaluation 3 to 50 minutes.

## Exact commands

These are the commands `complete_assignment.py` runs, in order.

```bash
# Tier 0: spiral, cosine and linear schedules
python train.py --tier 0 --steps 60000 --schedule cosine --seed 0
python eval.py --ckpt runs/tier0_cosine_seed0.pt --steps 3 5 10 20 50 200 --seed 0 --out runs/eval_tier0_cosine
python train.py --tier 0 --steps 60000 --schedule cosine --seed 1
python eval.py --ckpt runs/tier0_cosine_seed1.pt --steps 3 5 10 20 50 200 --seed 1 --out runs/eval_tier0_cosine
python train.py --tier 0 --steps 60000 --schedule cosine --seed 2
python eval.py --ckpt runs/tier0_cosine_seed2.pt --steps 3 5 10 20 50 200 --seed 2 --out runs/eval_tier0_cosine
python train.py --tier 0 --steps 60000 --schedule linear --seed 0
python eval.py --ckpt runs/tier0_linear_seed0.pt --steps 3 5 10 20 50 200 --seed 0 --out runs/eval_tier0_linear
python train.py --tier 0 --steps 60000 --schedule linear --seed 1
python eval.py --ckpt runs/tier0_linear_seed1.pt --steps 3 5 10 20 50 200 --seed 1 --out runs/eval_tier0_linear
python train.py --tier 0 --steps 60000 --schedule linear --seed 2
python eval.py --ckpt runs/tier0_linear_seed2.pt --steps 3 5 10 20 50 200 --seed 2 --out runs/eval_tier0_linear

# Tier 1 main: MNIST, lr 1e-3
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --seed 0 --out runs/main/seed0
python sample.py --ckpt runs/main/seed0/tier1_cosine_seed0.pt --steps 50 --w 1.0 --seed 0 --out runs/main/seed0/samples
python eval.py --ckpt runs/main/seed0/tier1_cosine_seed0.pt --steps 50 --w 0 1 2 3 5 8 --seed 0 --out runs/main/seed0/eval
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --seed 1 --out runs/main/seed1
python sample.py --ckpt runs/main/seed1/tier1_cosine_seed1.pt --steps 50 --w 1.0 --seed 1 --out runs/main/seed1/samples
python eval.py --ckpt runs/main/seed1/tier1_cosine_seed1.pt --steps 50 --w 0 1 2 3 5 8 --seed 1 --out runs/main/seed1/eval
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --seed 2 --out runs/main/seed2
python sample.py --ckpt runs/main/seed2/tier1_cosine_seed2.pt --steps 50 --w 1.0 --seed 2 --out runs/main/seed2/samples
python eval.py --ckpt runs/main/seed2/tier1_cosine_seed2.pt --steps 50 --w 0 1 2 3 5 8 --seed 2 --out runs/main/seed2/eval
```

Ablations change one setting from main. They are evaluated only at w = 0, 1, 3 to save time, except the lr 2e-4 runs, which were our first Tier 1 runs and were evaluated at all six w values.

```bash
# Ablation lr 2e-4 (train.py's own Tier 1 default; commands as run, with seed 0's defaults --seed 0 --out runs written out)
python train.py --tier 1 --epochs 20 --schedule cosine --seed 0 --out runs
python eval.py --ckpt runs/tier1_cosine_seed0.pt --steps 50 --w 0 1 2 3 5 8 --seed 0 --out runs/eval
python train.py --tier 1 --epochs 20 --schedule cosine --seed 1 --out runs/seed1
python eval.py --ckpt runs/seed1/tier1_cosine_seed1.pt --steps 50 --w 0 1 2 3 5 8 --seed 1 --out runs/seed1/eval
python train.py --tier 1 --epochs 20 --schedule cosine --seed 2 --out runs/seed2
python eval.py --ckpt runs/seed2/tier1_cosine_seed2.pt --steps 50 --w 0 1 2 3 5 8 --seed 2 --out runs/seed2/eval

# Ablation p_uncond0.2: main with --p_uncond 0.2
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --p_uncond 0.2 --seed 0 --out runs/abl/p_uncond0.2/seed0
python eval.py --ckpt runs/abl/p_uncond0.2/seed0/tier1_cosine_seed0.pt --steps 50 --w 0 1 3 --seed 0 --out runs/abl/p_uncond0.2/seed0/eval
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --p_uncond 0.2 --seed 1 --out runs/abl/p_uncond0.2/seed1
python eval.py --ckpt runs/abl/p_uncond0.2/seed1/tier1_cosine_seed1.pt --steps 50 --w 0 1 3 --seed 1 --out runs/abl/p_uncond0.2/seed1/eval
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --p_uncond 0.2 --seed 2 --out runs/abl/p_uncond0.2/seed2
python eval.py --ckpt runs/abl/p_uncond0.2/seed2/tier1_cosine_seed2.pt --steps 50 --w 0 1 3 --seed 2 --out runs/abl/p_uncond0.2/seed2/eval

# Ablation ep5: main with --epochs 5
python train.py --tier 1 --epochs 5 --schedule cosine --lr 0.001 --seed 0 --out runs/abl/ep5/seed0
python eval.py --ckpt runs/abl/ep5/seed0/tier1_cosine_seed0.pt --steps 50 --w 0 1 3 --seed 0 --out runs/abl/ep5/seed0/eval
python train.py --tier 1 --epochs 5 --schedule cosine --lr 0.001 --seed 1 --out runs/abl/ep5/seed1
python eval.py --ckpt runs/abl/ep5/seed1/tier1_cosine_seed1.pt --steps 50 --w 0 1 3 --seed 1 --out runs/abl/ep5/seed1/eval
python train.py --tier 1 --epochs 5 --schedule cosine --lr 0.001 --seed 2 --out runs/abl/ep5/seed2
python eval.py --ckpt runs/abl/ep5/seed2/tier1_cosine_seed2.pt --steps 50 --w 0 1 3 --seed 2 --out runs/abl/ep5/seed2/eval

# Ablation ep10: main with --epochs 10
python train.py --tier 1 --epochs 10 --schedule cosine --lr 0.001 --seed 0 --out runs/abl/ep10/seed0
python eval.py --ckpt runs/abl/ep10/seed0/tier1_cosine_seed0.pt --steps 50 --w 0 1 3 --seed 0 --out runs/abl/ep10/seed0/eval
python train.py --tier 1 --epochs 10 --schedule cosine --lr 0.001 --seed 1 --out runs/abl/ep10/seed1
python eval.py --ckpt runs/abl/ep10/seed1/tier1_cosine_seed1.pt --steps 50 --w 0 1 3 --seed 1 --out runs/abl/ep10/seed1/eval
python train.py --tier 1 --epochs 10 --schedule cosine --lr 0.001 --seed 2 --out runs/abl/ep10/seed2
python eval.py --ckpt runs/abl/ep10/seed2/tier1_cosine_seed2.pt --steps 50 --w 0 1 3 --seed 2 --out runs/abl/ep10/seed2/eval

# Ablation base64: main with --base 64
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --base 64 --seed 0 --out runs/abl/base64/seed0
python eval.py --ckpt runs/abl/base64/seed0/tier1_cosine_seed0.pt --steps 50 --w 0 1 3 --seed 0 --out runs/abl/base64/seed0/eval
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --base 64 --seed 1 --out runs/abl/base64/seed1
python eval.py --ckpt runs/abl/base64/seed1/tier1_cosine_seed1.pt --steps 50 --w 0 1 3 --seed 1 --out runs/abl/base64/seed1/eval
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --base 64 --seed 2 --out runs/abl/base64/seed2
python eval.py --ckpt runs/abl/base64/seed2/tier1_cosine_seed2.pt --steps 50 --w 0 1 3 --seed 2 --out runs/abl/base64/seed2/eval

# Ablation base128: main with --base 128
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --base 128 --seed 0 --out runs/abl/base128/seed0
python eval.py --ckpt runs/abl/base128/seed0/tier1_cosine_seed0.pt --steps 50 --w 0 1 3 --seed 0 --out runs/abl/base128/seed0/eval
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --base 128 --seed 1 --out runs/abl/base128/seed1
python eval.py --ckpt runs/abl/base128/seed1/tier1_cosine_seed1.pt --steps 50 --w 0 1 3 --seed 1 --out runs/abl/base128/seed1/eval
python train.py --tier 1 --epochs 20 --schedule cosine --lr 0.001 --base 128 --seed 2 --out runs/abl/base128/seed2
python eval.py --ckpt runs/abl/base128/seed2/tier1_cosine_seed2.pt --steps 50 --w 0 1 3 --seed 2 --out runs/abl/base128/seed2/eval
```

(`complete_assignment.py` would launch the lr 2e-4 runs as `--lr 0.0002` and evaluate them at w = 0, 1, 3 only. That is the same training configuration, since 2e-4 is `train.py`'s Tier 1 default, and the report uses only those three w values.)

Report figures and sampling-only experiments, after all of the above:

```bash
python report_figures.py
```

Settings are hardcoded at the top of the file. It reads the checkpoints in `runs/` and writes `results/figures/*.png` plus `results/figures/extra_numbers.json`. A figure whose PNG already exists is skipped; delete it to redraw.

## Where each reported number comes from

| Report item | Produced by | File |
|---|---|---|
| Tier 0 d vs steps, cosine vs linear (Table 1, Fig. 1) | `eval.py` (Tier 0) | `runs/eval_tier0_<schedule>/results_tier0_seed<s>.json` |
| Tier 1 metrics vs w (Table 2, Figs. 4, 5) | `eval.py` (Tier 1 main) | `runs/main/seed<s>/eval/results_tier1_seed<s>.json` |
| Uncurated 8×8 grid, per-epoch grids (Figs. 3, 7) | `sample.py`, `train.py` (main, seed 0) | `runs/main/seed0/samples/`, `runs/main/seed0/grids/` |
| Training time, loss curve | `train.py` | `*_train.json` next to each checkpoint |
| Sampling time at 50 steps | `sample.py` | `logs/sample_t1_main_s<s>.txt` |
| Ablation table (Table 4, Fig. 12) | `train.py` + `eval.py` (ablations) | `runs/.../eval/results_tier1_seed<s>.json` |
| Step sweep and cost (Table 3), timestep spacing (Table 5), guidance interval (Table 6), memorization, per-class match, EDA numbers | `report_figures.py` | `results/figures/extra_numbers.json` |
| All tables in one place | `complete_assignment.py` | `results/summary.md` |

`results/` collects copies of these files in the layout the course asks for (`results_tier0_<schedule>_seed<s>.json`, `results_tier1_seed<s>.json`, `grid_8x8.png`, `grids/ep*.png`, `*_train.json`, and `ablations/`).

## Seeds

`--seed s` sets the training seed, and the same `s` is passed to `sample.py` and `eval.py` for that run. `eval.py` always draws the 300 sevens used for the variety metric with seed 7, whatever `--seed` is. In `report_figures.py`, experiments over the three main models use each model's own seed; single illustrative figures and the variety samples use fixed seeds (0, 5, 7 or 11).

GPU results can differ slightly from run to run because some CUDA kernels are non-deterministic, so a rerun may not match the reported numbers to the last digit.
