# -*- coding: utf-8 -*-
"""중간 보고서용 그림 + 샘플링 전용 실험 두 개. complete_assignment.py 가 부릅니다 (단독 실행도 됩니다).

  results/figures/figNN_*.png      그림
  results/figures/extra_numbers.json   샘플링 실험의 숫자 (보고서 숫자의 출처)

모델에서 샘플을 뽑는 그림은 png 가 이미 있으면 건너뜁니다 (다시 만들려면 png 를 지우세요).
JSON 만 읽는 그림(학습 곡선, guidance trade-off, ablation 등)은 매번 다시 그립니다.

샘플링 실험 (학습 없이, 이미 있는 체크포인트로):
  A. 타임스텝 간격: 균등(timestep_schedule) vs 이차(t=0 근처에 촘촘)    Tier 0 d, Tier 1 fidelity
  B. guidance 구간: 모든 t 에서 guidance vs GUIDE_INTERVAL 안에서만      Tier 1 fidelity/variety/NN
두 실험 모두 ddim_step 은 그대로 쓰고, 그걸 엮는 루프(loop)만 바꿉니다. loop 는 기본 설정에서
sample_loop 과 같은 결과를 내는지 시작할 때 assert 로 확인합니다.
"""
import functools
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import data as D
import metrics
from diffusion import ddim_step, make_schedule, q_sample, timestep_schedule
from sample import load, sample_loop

# ---------------------------------------------------------------- 설정
SEEDS = [0, 1, 2]
RES = Path("results")
FIG = RES / "figures"
T0_SCHEDULES = ["cosine", "linear"]
T0_EVAL_STEPS = [3, 5, 10, 20, 50, 200]
T0_SCATTER_STEPS = [3, 10, 50, 200]
SPACING_STEPS_T0 = [3, 5, 10, 20, 50]          # 실험 A, Tier 0
SWEEP_STEPS_T1 = [5, 10, 20, 50, 100]          # 실험 A, Tier 1 (w=1)
GUIDE_W = [1, 3, 5, 8]                         # 실험 B
GUIDE_INTERVAL = (200, 700)                    # 이 t 구간에서만 guidance. 숫자 모양이 정해지는 구간 (slide 12: t=450 읽힘, 600 사라짐)
N_T1 = 500                                     # 실험용 Tier 1 샘플 수 (eval.py 는 1000)
N_T0 = 1000                                    # eval.py 와 같게 -> 균등 간격 d 가 eval.py 숫자와 일치해야 함
FORWARD_T = [0, 100, 200, 300, 450, 600, 800, 999]
TRAJ_IDX = [0, 10, 20, 30, 40, 44, 46, 48, 49]  # 50스텝 중 x0 예측을 보여줄 스텝
ABL = [("main", "lr 1e-3\n(main)"), ("lr2e-4", "lr 2e-4"), ("p_uncond0.2", "p_uncond\n0.2"),
       ("ep5", "5 ep"), ("ep10", "10 ep"), ("base64", "base 64"), ("base128", "base 128")]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

FIG.mkdir(parents=True, exist_ok=True)
EXTRA_PATH = FIG / "extra_numbers.json"
EXTRA = json.load(open(EXTRA_PATH, encoding="utf-8")) if EXTRA_PATH.exists() else {}


# ---------------------------------------------------------------- 공용
def main_ckpt(s):
    return Path("runs/main") / f"seed{s}" / f"tier1_cosine_seed{s}.pt"


@functools.lru_cache(None)
def mnist():
    xtr, ytr = D.mnist(train=True)
    xte, yte = D.mnist(train=False)
    return xtr, ytr, xte, yte


@functools.lru_cache(None)
def clf():
    torch.manual_seed(0)
    xtr, ytr, xte, yte = mnist()
    c, acc = metrics.train_classifier(xtr, ytr, xte, yte, DEV)
    EXTRA["figure_classifier_acc"] = acc
    return c


@functools.lru_cache(None)
def model_of(path):
    m, c = load(path, DEV)
    return m, make_schedule(c["T"], c["schedule"], DEV)[1], c["T"]


def spacing(T, n, kind):
    if kind == "uniform":
        return timestep_schedule(T, n)
    u = np.linspace(1, 0, n)                            # 이차: t = (T-1) u^2 -> t=0 근처가 촘촘
    return list(dict.fromkeys(np.round((T - 1) * u ** 2).astype(int).tolist()))


@torch.no_grad()
def loop(model, shape, ab, steps, y=None, w=0.0, seed=0, interval=None, keep=False):
    """sample_loop 과 같은 루프에 (1) 임의의 타임스텝 목록, (2) guidance 구간, (3) x0 기록을 더한 것."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(shape, device=DEV, generator=g)
    traj = []
    for i, t in enumerate(steps):
        tt = torch.full((len(x),), t, device=DEV, dtype=torch.long)
        if y is None:
            eps = model(x, tt)
        else:
            eps = model(x, tt, y)
            if w and (interval is None or interval[0] <= t <= interval[1]):
                eps = eps + w * (eps - model(x, tt, torch.full_like(y, model.NULL_CLASS)))
        if keep:
            a = ab[t]
            traj.append(((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1).cpu())
        x = ddim_step(x, eps, t, steps[i + 1] if i + 1 < len(steps) else None, ab)
    return (x, traj) if keep else x


@torch.no_grad()
def nn_dist(x, ref, chunk=200):
    """각 샘플에서 ref 까지 L2 최근접 거리와 인덱스 (metrics.nn_to_train 의 샘플별 버전)."""
    s, r = x.reshape(len(x), -1).to(DEV), ref.reshape(len(ref), -1).to(DEV)
    out = [torch.cdist(s[i:i + chunk], r).min(1) for i in range(0, len(s), chunk)]
    return torch.cat([o.values for o in out]).cpu(), torch.cat([o.indices for o in out]).cpu()


def labels(n):
    return torch.arange(n, device=DEV) % 10


def self_check():
    """기본 설정의 loop 가 채점 대상 sample_loop 과 같은 샘플을 내는지."""
    m, ab, T = model_of(main_ckpt(0))
    y = labels(16)
    a = sample_loop(m, (16, 1, 28, 28), ab, T, 10, y=y, w=1.0, seed=3, device=DEV)
    b = loop(m, (16, 1, 28, 28), ab, timestep_schedule(T, 10), y=y, w=1.0, seed=3)
    assert torch.allclose(a, b, atol=1e-5), "loop 이 sample_loop 과 다릅니다"
    m0, ab0, T0 = model_of(Path("runs/tier0_cosine_seed0.pt"))
    a = sample_loop(m0, (64, 2), ab0, T0, 10, seed=3, device=DEV)
    assert torch.allclose(a, loop(m0, (64, 2), ab0, timestep_schedule(T0, 10), seed=3), atol=1e-5)
    print("[check] loop == sample_loop  OK", flush=True)


def save(fig, name):
    for ax in fig.axes:                                  # 로그 축 보조눈금 라벨이 겹치는 것 방지
        if ax.get_xscale() == "log":
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG / name}", flush=True)


def jload(name):
    p = RES / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def mean_std(xs):
    xs = np.asarray(xs, float)
    return xs.mean(0), (xs.std(0, ddof=1) if len(xs) > 1 else np.zeros_like(xs.mean(0)))


def imgrid(ax, imgs, nrow):
    """(N,1,28,28) [-1,1] -> 한 장의 그리드 이미지."""
    x = ((imgs + 1) / 2).clamp(0, 1).squeeze(1).cpu().numpy()
    ncol = int(np.ceil(len(x) / nrow))
    canvas = np.ones((nrow * 30, ncol * 30))
    for k, im in enumerate(x):
        r, c = divmod(k, ncol)
        canvas[r * 30 + 1:r * 30 + 29, c * 30 + 1:c * 30 + 29] = im
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


# ================================================================ 1. 데이터셋 / 2. EDA
def fig01_mnist_examples():
    xtr, ytr, _, _ = mnist()
    imgs = torch.cat([xtr[ytr == k][:12] for k in range(10)])
    fig, ax = plt.subplots(figsize=(6, 5))
    imgrid(ax, imgs, 10)
    ax.set_ylabel("class 0 → 9"); ax.set_title("MNIST train examples (12 per class, uncurated)")
    save(fig, "fig01_mnist_examples.png")


def fig02_spiral():
    x, m = D.spiral(3000), D.true_manifold(4000)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(x[:, 0], x[:, 1], s=2, alpha=0.5, label="training data (n=3000, noise σ=0.005)")
    ax.plot(m[:2000, 0], m[:2000, 1], "k-", lw=0.8, label="true curve (used by d)")
    ax.plot(m[2000:, 0], m[2000:, 1], "k-", lw=0.8)
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="lower left"); ax.set_title("Tier 0: two-arm spiral")
    save(fig, "fig02_spiral.png")


def fig03_class_counts():
    _, ytr, _, yte = mnist()
    tr, te = torch.bincount(ytr, minlength=10), torch.bincount(yte, minlength=10)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    for a, c, name in [(ax[0], tr, "train (60,000)"), (ax[1], te, "test (10,000)")]:
        a.bar(range(10), c.numpy(), color="tab:blue")
        a.axhline(c.float().mean(), ls="--", color="gray", lw=1)
        a.set_xticks(range(10)); a.set_title(f"{name}: min {int(c.min())} / max {int(c.max())}  "
                                             f"(ratio {float(c.max() / c.min()):.2f})", fontsize=9)
    ax[0].set_ylabel("count")
    save(fig, "fig03_class_counts.png")


def fig04_pixel_hist():
    xtr, *_ = mnist()
    v = xtr.flatten()
    bg = float((v <= -0.999).float().mean())
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(v.numpy(), bins=64, color="tab:blue", log=True)
    ax.set_xlabel("pixel value after Normalize((0.5,),(0.5,))  [-1, 1]"); ax.set_ylabel("count (log)")
    ax.set_title(f"{bg:.1%} of pixels are exactly background (-1)", fontsize=9)
    EXTRA["pixel_background_fraction"] = bg
    save(fig, "fig04_pixel_hist.png")


def fig05_class_mean_std():
    xtr, ytr, _, _ = mnist()
    fig, ax = plt.subplots(2, 10, figsize=(10, 2.4))
    for k in range(10):
        xk = xtr[ytr == k]
        ax[0, k].imshow(xk.mean(0)[0], cmap="gray"); ax[1, k].imshow(xk.std(0)[0], cmap="magma")
        ax[0, k].set_title(str(k)); ax[0, k].axis("off"); ax[1, k].axis("off")
    ax[0, 0].text(-8, 14, "mean", rotation=90, va="center"); ax[1, 0].text(-8, 14, "std", rotation=90, va="center")
    save(fig, "fig05_class_mean_std.png")


def fig06_within_class_spread():
    _, _, xte, yte = mnist()
    v = [metrics.mean_pairwise(xte[yte == k], DEV) for k in range(10)]
    EXTRA["real_within_class_pairwise_test"] = v
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(10), v, color=["tab:orange" if k == 7 else "tab:blue" for k in range(10)])
    ax.axhline(np.mean(v), ls="--", color="gray", lw=1, label=f"mean {np.mean(v):.2f}")
    ax.set_xticks(range(10)); ax.set_ylabel("mean pairwise L2 (test)")
    ax.set_title("Within-class spread of real digits (orange = 7, used for variety)", fontsize=9); ax.legend()
    save(fig, "fig06_within_class_spread.png")


def fig07_forward_process():
    _, _, xte, _ = mnist()
    x0 = xte[:1].to(DEV)
    noise = torch.randn(x0.shape, generator=torch.Generator(device=DEV).manual_seed(0), device=DEV)
    fig = plt.figure(figsize=(13, 3.6))
    left, right = fig.subfigures(1, 2, width_ratios=[2.3, 1])
    ax = left.subplots(2, len(FORWARD_T))
    for r, kind in enumerate(["cosine", "linear"]):
        ab = make_schedule(1000, kind, DEV)[1]
        for c, t in enumerate(FORWARD_T):
            xt = q_sample(x0, torch.tensor([t], device=DEV), noise, ab)
            ax[r, c].imshow(xt[0, 0].cpu(), cmap="gray", vmin=-1.5, vmax=1.5); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f"t={t}", fontsize=8)
        ax[r, 0].set_ylabel(kind)
    left.suptitle("Forward process q(x_t | x_0), same x_0 and noise", fontsize=10)
    a1, a2 = right.subplots(2, 1, sharex=True)
    for kind in ["cosine", "linear"]:
        ab = make_schedule(1000, kind, "cpu")[1]
        a1.plot(ab, label=kind); a2.plot(torch.log(ab / (1 - ab)), label=kind)
    for a in (a1, a2):
        a.axvspan(450, 600, color="gray", alpha=0.15)
    a1.set_ylabel("ᾱ_t"); a2.set_ylabel("log SNR"); a2.set_xlabel("t"); a1.legend(fontsize=8)
    a1.set_title("shaded: digit disappears (cosine)", fontsize=8)
    save(fig, "fig07_forward_process.png")


# ================================================================ 4. 결과
def fig08_t0_samples():
    man = D.true_manifold().to(DEV)
    ref = D.true_manifold(4000)
    fig, ax = plt.subplots(2, len(T0_SCATTER_STEPS), figsize=(3 * len(T0_SCATTER_STEPS), 6), sharex=True, sharey=True)
    for r, sch in enumerate(T0_SCHEDULES):
        m, ab, T = model_of(Path(f"runs/tier0_{sch}_seed0.pt"))
        for c, n in enumerate(T0_SCATTER_STEPS):
            x = loop(m, (2000, 2), ab, timestep_schedule(T, n), seed=0)
            d = metrics.distance_to_manifold(x, man, DEV)
            a = ax[r, c]
            a.plot(ref[:2000, 0], ref[:2000, 1], "k-", lw=0.5); a.plot(ref[2000:, 0], ref[2000:, 1], "k-", lw=0.5)
            a.scatter(*x.cpu().T, s=1, alpha=0.4, color="tab:blue" if sch == "cosine" else "tab:orange")
            a.set_title(f"{sch}, {n} steps\nd = {d:.4f}", fontsize=9); a.set_aspect("equal")
            a.set_xlim(-1.2, 1.2); a.set_ylim(-1.2, 1.2)
    fig.suptitle("Tier 0 samples (seed 0, n=2000) over the true curve", y=1.03)
    save(fig, "fig08_t0_samples.png")


def fig09_d_curve():
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for sch, c in zip(T0_SCHEDULES, ["tab:blue", "tab:orange"]):
        rs = [jload(f"results_tier0_{sch}_seed{s}.json") for s in SEEDS]
        mu, sd = mean_std([[r["steps"][str(n)]["d"] for n in T0_EVAL_STEPS] for r in rs])
        ax.errorbar(T0_EVAL_STEPS, mu, yerr=sd, marker="o", capsize=3, color=c, label=sch)
    floor = jload("results_tier0_cosine_seed0.json")["floor_real_heldout"]
    ax.axhline(floor, ls="--", color="gray", label=f"held-out real data ({floor:.4f})")
    ax.set(xscale="log", yscale="log", xlabel="DDIM sampling steps", ylabel="d (mean distance to curve)",
           title=f"Tier 0: steps vs d (mean ± std, {len(SEEDS)} seeds)")
    ax.set_xticks(T0_EVAL_STEPS, [str(n) for n in T0_EVAL_STEPS]); ax.legend(); ax.grid(alpha=0.3, which="both")
    save(fig, "fig09_d_curve.png")


def fig10_t0_spacing():
    """실험 A (Tier 0): 균등 vs 이차 타임스텝 간격."""
    man = D.true_manifold().to(DEV)
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, sch in zip(axs, T0_SCHEDULES):
        out[sch] = {}
        for kind, ls in [("uniform", "-"), ("quadratic", "--")]:
            ds = []
            for s in SEEDS:
                m, ab, T = model_of(Path(f"runs/tier0_{sch}_seed{s}.pt"))
                ds.append([metrics.distance_to_manifold(loop(m, (N_T0, 2), ab, spacing(T, n, kind), seed=s), man, DEV)
                           for n in SPACING_STEPS_T0])
            out[sch][kind] = dict(zip(map(str, SPACING_STEPS_T0), map(list, zip(*ds))))
            mu, sd = mean_std(ds)
            ax.errorbar(SPACING_STEPS_T0, mu, yerr=sd, ls=ls, marker="o", capsize=3, label=kind)
        ev = [jload(f"results_tier0_{sch}_seed{s}.json")["steps"] for s in SEEDS]   # 균등 = eval.py 와 같은 조건
        out[sch]["max_abs_diff_vs_eval_py"] = max(abs(out[sch]["uniform"][str(n)][i] - ev[i][str(n)]["d"])
                                                  for n in SPACING_STEPS_T0 for i in range(len(SEEDS)))
        ax.set(xscale="log", yscale="log", xlabel="sampling steps", title=f"{sch} schedule")
        ax.set_xticks(SPACING_STEPS_T0, [str(n) for n in SPACING_STEPS_T0]); ax.grid(alpha=0.3, which="both"); ax.legend()
    axs[0].set_ylabel("d (mean ± std, 3 seeds)")
    fig.suptitle("Experiment A: uniform vs quadratic timestep spacing (Tier 0)", y=1.03)
    EXTRA["A_t0_spacing_d"] = out
    save(fig, "fig10_t0_spacing.png")


def fig11_loss_vs_grids():
    tr = [jload(f"tier1_cosine_seed{s}_train.json") for s in SEEDS]
    mu, sd = mean_std([t["losses"] for t in tr])
    ep = np.arange(1, len(mu) + 1)
    show = [e for e in [1, 2, 5, 10, 20] if e <= len(mu)]
    fig, ax = plt.subplot_mosaic([["loss"] + [f"e{e}" for e in show]], figsize=(15, 3),
                                 width_ratios=[1.6] + [1] * len(show))
    ax["loss"].plot(ep, mu, marker="o", ms=3); ax["loss"].fill_between(ep, mu - sd, mu + sd, alpha=0.3)
    ax["loss"].set(xlabel="epoch", ylabel="train MSE (mean ± std)", title="loss flattens early…")
    for e in show:
        p = Path("runs/main/seed0/grids") / f"ep{e:02d}.png"
        if p.exists():
            ax[f"e{e}"].imshow(plt.imread(p))
        ax[f"e{e}"].set_title(f"epoch {e} (loss {mu[e - 1]:.4f})", fontsize=9); ax[f"e{e}"].axis("off")
    fig.suptitle("…while the fixed-seed sample grid keeps improving (seed 0, w=1, 50 steps)", y=1.03)
    save(fig, "fig11_loss_vs_grids.png")


def fig12_guidance_tradeoff():
    rs = [jload(f"results_tier1_seed{s}.json") for s in SEEDS]
    ws = sorted(float(w) for w in rs[0]["w"])
    g = lambda k: mean_std([[r["w"][str(w)][k] for w in ws] for r in rs])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for k, lab in [("fidelity", "fidelity"), ("class_match", "class match")]:
        mu, sd = g(k); ax[0].errorbar(ws, mu, yerr=sd, marker="o", capsize=3, label=lab)
    mu, sd = g("variety_ratio"); ax[1].errorbar(ws, mu, yerr=sd, marker="o", capsize=3, color="tab:green")
    ax[1].axhline(1, ls="--", color="gray", lw=1, label="real data")
    mu, sd = g("nn_to_train"); ax[2].errorbar(ws, mu, yerr=sd, marker="o", capsize=3, color="tab:red", label="generated")
    real = np.mean([r["real"]["nn_to_train"] for r in rs])
    ax[2].axhline(real, ls="--", color="gray", label=f"real held-out ({real:.2f})")
    for a, t in zip(ax, ["fidelity (↑)", "within-class variety vs real (↑)", "NN-to-train (must stay above real)"]):
        a.set_xlabel("guidance w"); a.set_title(t, fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle(f"Guidance trade-off (main, mean ± std over {len(rs)} seeds)", y=1.03)
    save(fig, "fig12_guidance_tradeoff.png")


def fig13_coverage():
    rs = [jload(f"results_tier1_seed{s}.json") for s in SEEDS]
    ws = sorted(float(w) for w in rs[0]["w"])
    h = np.mean([[r["w"][str(w)]["coverage_hist"] for w in ws] for r in rs], 0)
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(h, cmap="viridis", vmin=0, vmax=0.2, aspect="auto")
    ax.set_yticks(range(len(ws)), [f"w={w:g}" for w in ws]); ax.set_xticks(range(10)); ax.set_xlabel("predicted class")
    fig.colorbar(im, label="fraction"); ax.set_title("Coverage: predicted-class histogram stays flat (0.1) at every w", fontsize=9)
    save(fig, "fig13_coverage.png")


def fig14_trajectory():
    m, ab, T = model_of(main_ckpt(0))
    steps = timestep_schedule(T, 50)
    x, traj = loop(m, (10, 1, 28, 28), ab, steps, y=labels(10), w=1.0, seed=5, keep=True)
    fig, ax = plt.subplots(10, len(TRAJ_IDX) + 1, figsize=(1.1 * (len(TRAJ_IDX) + 1), 11))
    for c, i in enumerate(TRAJ_IDX + [None]):
        for r in range(10):
            im = traj[i][r, 0] if i is not None else x[r, 0].cpu()
            ax[r, c].imshow(im, cmap="gray", vmin=-1, vmax=1); ax[r, c].axis("off")
        ax[0, c].set_title(f"x̂0 @ t={steps[i]}" if i is not None else "final", fontsize=7)
    fig.suptitle("Denoising trajectory: predicted clean image x̂0 at each DDIM step (50 steps, w=1)", fontsize=10, y=1.03)
    save(fig, "fig14_trajectory.png")


def fig15_guidance_artifacts():
    m, ab, T = model_of(main_ckpt(0))
    fig, ax = plt.subplots(3, 1, figsize=(8, 3.4))
    for a, w in zip(ax, [0.0, 3.0, 8.0]):
        x = sample_loop(m, (10, 1, 28, 28), ab, T, 50, y=labels(10), w=w, seed=11, device=DEV)
        imgrid(a, x, 1); a.set_ylabel(f"w={w:g}", rotation=0, labelpad=20)
    fig.suptitle("Same noise, same labels, only w changes (seed 11)", fontsize=10, y=1.03)
    save(fig, "fig15_guidance_artifacts.png")


def fig16_confusion_w0():
    m, ab, T = model_of(main_ckpt(0))
    y = labels(1000)
    x = sample_loop(m, (1000, 1, 28, 28), ab, T, 50, y=y, w=0.0, seed=0, device=DEV)   # eval.py 의 w=0 과 같은 조건
    pred = metrics._probs(clf(), x, DEV).argmax(-1)
    cm = torch.zeros(10, 10)
    for a, b in zip(y.cpu(), pred):
        cm[a, b] += 1
    cm = cm / cm.sum(1, keepdim=True)
    EXTRA["per_class_match_w0_seed0"] = cm.diag().tolist()
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set(xlabel="classifier prediction", ylabel="requested class", xticks=range(10), yticks=range(10))
    ax.set_title("Requested vs predicted class at w=0 (seed 0, n=1000)", fontsize=9)
    save(fig, "fig16_confusion_w0.png")


def fig17_memorization():
    xtr, _, xte, _ = mnist()
    m, ab, T = model_of(main_ckpt(0))
    x = sample_loop(m, (1000, 1, 28, 28), ab, T, 50, y=labels(1000), w=1.0, seed=0, device=DEV).cpu()
    dg, ig = nn_dist(x, xtr)
    dr, _ = nn_dist(xte[:1000], xtr)
    EXTRA["memorization_seed0_w1"] = {"gen_mean": float(dg.mean()), "real_mean": float(dr.mean()),
                                      "gen_min": float(dg.min()), "real_min": float(dr.min()),
                                      "gen_frac_below_real_p5": float((dg < dr.quantile(0.05)).float().mean())}
    fig = plt.figure(figsize=(12, 3.4))
    l, r = fig.subfigures(1, 2, width_ratios=[1, 1.4])
    a = l.subplots()
    bins = np.linspace(0, float(max(dg.max(), dr.max())), 50)
    a.hist(dr.numpy(), bins, alpha=0.6, label=f"real test (mean {dr.mean():.2f})")
    a.hist(dg.numpy(), bins, alpha=0.6, label=f"generated w=1 (mean {dg.mean():.2f})")
    a.set(xlabel="L2 distance to nearest training image", ylabel="count"); a.legend(fontsize=8)
    order = dg.argsort()[:8]                               # 가장 '베낀 것처럼' 보이는 8개
    b = r.subplots(2, 1)
    imgrid(b[0], x[order], 1); imgrid(b[1], xtr[ig[order]], 1)
    b[0].set_ylabel("generated", fontsize=8); b[1].set_ylabel("nearest train", fontsize=8)
    r.suptitle("8 closest generated samples and their nearest training image", fontsize=9)
    save(fig, "fig17_memorization.png")


def fig18_t1_step_sweep():
    """실험 A (Tier 1): 스텝 수 x 간격, w=1. fidelity 와 비용."""
    res = {}
    for kind in ["uniform", "quadratic"]:
        res[kind] = {"fidelity": [], "class_match": [], "sec_per_100": []}
        for s in SEEDS:
            m, ab, T = model_of(main_ckpt(s))
            f, cmatch, sec = [], [], []
            for n in SWEEP_STEPS_T1:
                y = labels(N_T1)
                if DEV == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                x = loop(m, (N_T1, 1, 28, 28), ab, spacing(T, n, kind), y=y, w=1.0, seed=s)
                if DEV == "cuda":
                    torch.cuda.synchronize()
                sec.append((time.time() - t0) / (N_T1 / 100))
                f.append(metrics.fidelity(clf(), x, DEV)); cmatch.append(metrics.class_match(clf(), x, y.cpu(), DEV))
            for k, v in zip(["fidelity", "class_match", "sec_per_100"], [f, cmatch, sec]):
                res[kind][k].append(v)
    EXTRA["A_t1_step_sweep_w1"] = {"steps": SWEEP_STEPS_T1, **res}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for kind, ls in [("uniform", "-"), ("quadratic", "--")]:
        mu, sd = mean_std(res[kind]["fidelity"]); ax[0].errorbar(SWEEP_STEPS_T1, mu, yerr=sd, ls=ls, marker="o", capsize=3, label=kind)
    mu, sd = mean_std(res["uniform"]["sec_per_100"]); ax[1].errorbar(SWEEP_STEPS_T1, mu, yerr=sd, marker="o", capsize=3)
    ax[0].set(xscale="log", xlabel="sampling steps", ylabel="fidelity", title="fidelity vs steps (w=1, 3 seeds)")
    ax[1].set(xscale="log", xlabel="sampling steps", ylabel="sec / 100 samples", title="cost vs steps (w=1 → 2 passes/step)")
    for a in ax:
        a.set_xticks(SWEEP_STEPS_T1, [str(n) for n in SWEEP_STEPS_T1]); a.grid(alpha=0.3)
    ax[0].legend()
    save(fig, "fig18_t1_step_sweep.png")


def fig19_guidance_interval():
    """실험 B: 모든 t 에서 guidance vs GUIDE_INTERVAL 안에서만."""
    xtr, _, xte, yte = mnist()
    real7 = xte[yte == 7][:300]
    res = {}
    for mode, itv in [("all_t", None), ("interval", GUIDE_INTERVAL)]:
        res[mode] = {k: [] for k in ["fidelity", "variety", "nn_to_train"]}
        for s in SEEDS:
            m, ab, T = model_of(main_ckpt(s))
            steps = timestep_schedule(T, 50)
            f, v, nn = [], [], []
            for w in GUIDE_W:
                x = loop(m, (N_T1, 1, 28, 28), ab, steps, y=labels(N_T1), w=w, seed=s, interval=itv)
                s7 = loop(m, (300, 1, 28, 28), ab, steps, y=torch.full((300,), 7, device=DEV), w=w, seed=7, interval=itv)
                f.append(metrics.fidelity(clf(), x, DEV))
                v.append(metrics.within_class_variety(s7, real7, DEV)[0])
                nn.append(metrics.nn_to_train(x, xtr, DEV))
            for k, val in zip(["fidelity", "variety", "nn_to_train"], [f, v, nn]):
                res[mode][k].append(val)
    EXTRA["B_guidance_interval"] = {"w": GUIDE_W, "interval_t": GUIDE_INTERVAL, **res}
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for mode, ls in [("all_t", "-"), ("interval", "--")]:
        lab = "guidance at all t" if mode == "all_t" else f"guidance only for t∈{list(GUIDE_INTERVAL)}"
        for a, k in zip(ax, ["fidelity", "variety", "nn_to_train"]):
            mu, sd = mean_std(res[mode][k]); a.errorbar(GUIDE_W, mu, yerr=sd, ls=ls, marker="o", capsize=3, label=lab)
    for a, t in zip(ax, ["fidelity (↑)", "variety of 7s vs real (↑)", "NN-to-train"]):
        a.set_xlabel("guidance w"); a.set_title(t, fontsize=10); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Experiment B: limiting guidance to an interval of t (n={N_T1}, 3 seeds)", y=1.03)
    save(fig, "fig19_guidance_interval.png")


def fig20_interval_samples():
    m, ab, T = model_of(main_ckpt(0))
    steps = timestep_schedule(T, 50)
    fig, ax = plt.subplots(2, 1, figsize=(8, 2.4))
    for a, itv, lab in [(ax[0], None, "w=8, all t"), (ax[1], GUIDE_INTERVAL, f"w=8, t∈{list(GUIDE_INTERVAL)}")]:
        imgrid(a, loop(m, (10, 1, 28, 28), ab, steps, y=labels(10), w=8.0, seed=11, interval=itv), 1)
        a.set_ylabel(lab, rotation=0, labelpad=45, fontsize=8)
    fig.suptitle("Experiment B samples (same noise as fig15)", fontsize=10, y=1.03)
    save(fig, "fig20_interval_samples.png")


def fig21_ablations():
    rows = []
    for cfg, lab in ABL:
        ev = [jload(f"results_tier1_seed{s}.json" if cfg == "main" else f"ablations/{cfg}_seed{s}.json") for s in SEEDS]
        tr = [jload(f"tier1_cosine_seed{s}_train.json" if cfg == "main" else f"ablations/{cfg}_seed{s}_train.json") for s in SEEDS]
        ev, tr = [e for e in ev if e], [t for t in tr if t]
        if ev:
            rows.append((lab, ev, tr))
    if len(rows) < 2:
        print("  (ablation 결과가 아직 부족해서 fig21 을 건너뜁니다)")
        return
    panels = [("fidelity @ w=0", lambda e: e["w"]["0.0"]["fidelity"]),
              ("fidelity @ w=1", lambda e: e["w"]["1.0"]["fidelity"]),
              ("variety @ w=1", lambda e: e["w"]["1.0"]["variety_ratio"]),
              ("NN-to-train @ w=1", lambda e: e["w"]["1.0"]["nn_to_train"]),
              ("train minutes", None)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), gridspec_kw={"hspace": 0.6})
    ax = axs.flat
    axs[1, 2].axis("off")
    for a, (title, f) in zip(ax, panels):
        vals = [[t["train_minutes"] for t in tr] if f is None else [f(e) for e in ev] for _, ev, tr in rows]
        mu = [np.mean(v) for v in vals]
        sd = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals]
        a.bar(range(len(rows)), mu, yerr=sd, capsize=3, color=["tab:blue"] + ["tab:gray"] * (len(rows) - 1))
        a.set_xticks(range(len(rows)), [f"{lab}\n(n={len(ev)})" for lab, ev, _ in rows], fontsize=8)
        a.set_title(title, fontsize=10)
        if f is not None and title.startswith("fid"):
            a.set_ylim(min(mu) - 0.05, 1.0)
        if title.startswith("variety"):
            a.set_ylim(0.9, 1.0)
        if title.startswith("NN"):
            real = np.mean([e["real"]["nn_to_train"] for e in rows[0][1]])
            a.axhline(real, ls="--", color="red", lw=1, label="real held-out"); a.set_ylim(real - 1, max(mu) + 1); a.legend(fontsize=7)
    fig.suptitle("Tier 1 ablations: one change from main each (mean ± std over seeds)", y=1.03)
    save(fig, "fig21_ablations.png")


# ---------------------------------------------------------------- 실행
SAMPLING = [fig01_mnist_examples, fig02_spiral, fig03_class_counts, fig04_pixel_hist, fig05_class_mean_std,
            fig06_within_class_spread, fig07_forward_process, fig08_t0_samples, fig10_t0_spacing,
            fig14_trajectory, fig15_guidance_artifacts, fig16_confusion_w0, fig17_memorization,
            fig18_t1_step_sweep, fig19_guidance_interval, fig20_interval_samples]
FROM_JSON = [fig09_d_curve, fig11_loss_vs_grids, fig12_guidance_tradeoff, fig13_coverage, fig21_ablations]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)
    failed = []
    try:
        self_check()
    except Exception:
        traceback.print_exc(); failed.append("self_check")
    for f in SAMPLING + FROM_JSON:
        if f in SAMPLING and any(FIG.glob(f.__name__ + "*.png")):
            print(f"[skip] {f.__name__}")
            continue
        print(f"[fig ] {f.__name__}", flush=True)
        try:
            f()
        except Exception:
            traceback.print_exc(); failed.append(f.__name__)
        json.dump(EXTRA, open(EXTRA_PATH, "w", encoding="utf-8"), indent=2)
    print("\n실패: " + ", ".join(failed) if failed else "\n그림 전부 완료")
    sys.exit(1 if failed else 0)
