# Diffusion toy project: 2D spiral → MNIST

CAS2250 Introduction to AI, task 2. DDPM-style diffusion trained from scratch on a 2D spiral (Tier 0) and on class-conditional MNIST with classifier-free guidance (Tier 1), sampled with DDIM.

Every number in the report comes from the commands in this README. Runs used seeds **0, 1 and 2**, and reported values are mean ± standard deviation over those three seeds unless the table below marks a result as single-seed.

## What we changed from the starter code

| File | Change |
|---|---|
| `train.py` | Filled in `training_step` (TODO 1). Added `--p_uncond` (default 0.1) and `--base` (default 96) options for the ablations; defaults keep the starter behaviour. Checkpoints and `*_train.json` also record `lr`, `p_uncond`, `base`, `epochs`. |
| `sample.py` | Filled in `sample_loop` (TODO 2). `load()` reads the model width from the checkpoint (`base`, default 96). |
| `complete_assignment.py` | New. Runs every command below in order, then collects `results/` and writes `results/summary.md`. |
| `report_figures.py` | New. Report figures and the two sampling-only experiments. |
| `eda.py` | New. Section 2 dataset statistics and Figure 1. |
| `data.py`, `diffusion.py`, `models.py`, `metrics.py`, `eval.py` | Unchanged. |

The lr 2e-4 runs were our first Tier 1 runs and predate the new fields, so their `*_train.json` files have no `lr`, `p_uncond`, `base` or `epochs` entries. Their values were 2e-4, 0.1, 96 and 20.

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

Section 2 dataset statistics and Figure 1 (needs only MNIST, no training):

```bash
python eda.py
```

Report figures and sampling-only experiments, after all of the above:

```bash
python report_figures.py
```

Settings are hardcoded at the top of the file. It reads the checkpoints in `runs/` and writes `results/figures/*.png` plus `results/figures/extra_numbers.json`. A figure whose PNG already exists is skipped; delete it to redraw.

## Where each reported number comes from

Table and figure numbers follow the submitted report (version of 2026-10-22); update this table if the report is renumbered. Paths are inside `results/`, which `complete_assignment.py` fills from `runs/`. `results/` is included in the submission zip, not in this repository.

| Report item | Seeds | Produced by | Numbers in | Figure file (`results/figures/`) |
|---|---|---|---|---|
| Section 2, Fig. 1, Tables 1–2: class counts and percentages, count std (population), pixel mean and std on [0, 1], zero-pixel share, size check, all-black and duplicate checks, foreground share and median width (pixel > 127) | n/a (Fig. 1: seed 2250) | `eda.py` | `eda.json` | Fig. 1 `fig00_mnist_samples.png` |
| Table 5, Figs. 2–3: Tier 0 d vs steps, cosine vs linear, sampling time | 3 (Fig. 3: seed 0) | `eval.py` (Tier 0); Fig. 3 by `report_figures.py` | `results_tier0_<schedule>_seed<s>.json` | Fig. 2 `fig09_d_curve.png`, Fig. 3 `fig08_t0_samples.png` |
| Table 6, Figs. 5–6: Tier 1 metrics vs w, real-test fidelity | 3 | `eval.py` (Tier 1 main) | `results_tier1_seed<s>.json` | Fig. 5 `fig12_guidance_tradeoff.png`, Fig. 6 `fig13_coverage.png` |
| Fig. 4: uncurated 8×8 grid | 1 (seed 0) | `sample.py` (main) | | `../grid_8x8.png` |
| Fig. 7: same noise, w = 0, 3, 8 | 1 (seed-0 model, noise seed 11) | `report_figures.py` | | `fig15_guidance_artifacts.png` |
| Table 7, Fig. 8: steps vs fidelity and cost, incl. the 50-step time of 5.06 s / 100 samples | 3 | `report_figures.py` | `figures/extra_numbers.json` → `A_t1_step_sweep_w1` | `fig18_t1_step_sweep.png` |
| Table 8, Fig. 9: ablations | 3 | `train.py` + `eval.py` (ablations) | `ablations/<setting>_seed<s>.json`, `ablations/<setting>_seed<s>_train.json` | `fig21_ablations.png` |
| Fig. 10: loss curve and per-epoch grids | loss 3, grids 1 (seed 0) | `train.py` (main) | `tier1_cosine_seed<s>_train.json` | `fig11_loss_vs_grids.png`, `../grids/ep*.png` |
| §4.3, Fig. 11: memorization check | 1 (seed 0, w = 1) | `report_figures.py` | `extra_numbers.json` → `memorization_seed0_w1` | `fig17_memorization.png` |
| Fig. 12: per-class match at w = 0 | 1 (seed 0) | `report_figures.py` | `extra_numbers.json` → `per_class_match_w0_seed0` | `fig16_confusion_w0.png` |
| Table 9, Fig. 13: timestep spacing (Tier 0) | 3 | `report_figures.py` | `extra_numbers.json` → `A_t0_spacing_d` | `fig10_t0_spacing.png` |
| Table 10, Figs. 14–15: guidance interval | 3 (Fig. 15: seed-0 model, noise seed 11) | `report_figures.py` | `extra_numbers.json` → `B_guidance_interval` | `fig19_guidance_interval.png`, `fig20_interval_samples.png` |
| Training time | 3 | `train.py` | `*_train.json` (`train_minutes`) | |
| Figs. A1–A7 and the two EDA numbers we compute (background-pixel fraction 0.809, within-class spread) | n/a | `report_figures.py` | `extra_numbers.json` → `pixel_background_fraction`, `real_within_class_pairwise_test` | `fig01`–`fig07` |
| All tables in one place | | `complete_assignment.py` | `summary.md` | |

`fig14_trajectory.png` is also generated but is not numbered in the final report.

### Evaluation protocols

Two different classifiers and sample counts are involved, so values for the same setting can differ slightly between tables (for example fidelity at w = 1 is 0.997 in Table 6 and 0.995 in Tables 7 and 10).

| Result | Samples per model seed | Classifier |
|---|---|---|
| Table 6, Figs. 5–6, Table 8 | 1,000 (variety: a separate 300 sevens) | the one `eval.py` trains for each run (98.58 ± 0.11% test accuracy). Variety and NN-to-train use no classifier. |
| Table 7, Fig. 8 | 500 | a separate classifier trained once in `report_figures.py` (`torch.manual_seed(0)`, 98.69% test accuracy, `figure_classifier_acc`) |
| Table 10, Fig. 14 | 500 for fidelity and NN-to-train, a separate 300 sevens for variety | same `report_figures.py` classifier, for fidelity only |
| Fig. 12 (per-class match) | 1,000, seed-0 model | same `report_figures.py` classifier |
| Table 9, Fig. 13 (spacing) | 1,000 2D points | none (distance to the true curve) |
| §4.3, Fig. 11 (memorization) | 1,000 generated and 1,000 test images, seed-0 model | none (pixel L2 nearest neighbour) |

The real-data reference rows (real-test fidelity 0.947 ± 0.007, NN-to-train 8.48) use the first 1,000 test images, scored by each main run's `eval.py` classifier. The spread across seeds comes from the classifier, not from different test images.

### Timing notes

The 50-step sampling time used in the report, 5.06 s per 100 samples, comes from `report_figures.py`: 500 samples per seed, with CUDA synchronization before and after the timer. `sample.py` also prints a time per 100 samples, but it generates only 64 samples, with no warm-up or synchronization, so we do not use that number.

The Tier 0 `sec_per_100` values come from `eval.py`, which times the step counts in order without a warm-up run or CUDA synchronization. The 3-step value (0.0058 s) is higher than the 5-step value, most likely because it includes first-call overhead; we do not read it as a stable GPU time.

## Seeds

`--seed s` sets the training seed, and the same `s` is passed to `sample.py` and `eval.py` for that run. `eval.py` always draws the 300 sevens used for the variety metric with seed 7, whatever `--seed` is. In `report_figures.py`, experiments over the three main models use each model's own seed; single illustrative figures and the variety samples use fixed seeds (0, 5, 7 or 11).

GPU results can differ slightly from run to run because some CUDA kernels are non-deterministic, so a rerun may not match the reported numbers to the last digit.

## Future Work

These are planned extensions, not completed results. The baseline settings and commands above stay as they are.

We plan to reduce the repeated computation of the current 50-step DDIM sampler and then extend digit generation to personalized, character-level handwriting. We will investigate teacher–student distillation using our trained diffusion model as the initial teacher. Consistency Models is a candidate approach, not a finalized implementation. One-step sampling will be compared with the 50-step baseline and lower-step DDIM configurations using quality, diversity, nearest-neighbor, and cost measurements. Setting the existing sampler's step count to 1 is not a one-step student, and fewer network calls do not by themselves mean a proportional wall-clock speedup.

For the handwriting task, a source glyph image will specify character content, and reference handwriting samples will specify the target style. The output will be an image of the same character in that style. We will keep content and style fixed while changing the sampling noise to investigate natural variation across repeated characters. One-shot style conditioning (one reference sample) and one-step inference (one network evaluation) are separate requirements.

| Planned input or output | Meaning |
|---|---|
| Content input | a source glyph image of one character |
| Style condition | a representation of the target style, taken from a set of reference handwriting samples |
| Sampling input | fresh Gaussian noise for each generated character |
| Output | a glyph image of the same character in the target style |
| Repeated characters | same content and style, different noise for each occurrence |

An interactive typing interface is optional and will depend on measured single-character latency. Initial outputs will be glyph images rather than an assumed drop-in replacement for an existing font engine. Dataset selection, architecture, pretrained-component use, and novelty remain to be finalized.
