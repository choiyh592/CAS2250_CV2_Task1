# -*- coding: utf-8 -*-
"""학습. 여기 TODO 하나가 여러분이 채울 부분입니다.

    python train.py --tier 0 --steps 60000          # 2차원 나선
    python train.py --tier 1 --epochs 20            # 28x28 MNIST

TODO를 채우기 전에 실행하면 무엇을 채워야 하는지 알려 주고 멈춥니다.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

import data as D
import models as M
from diffusion import make_schedule, q_sample, check_data


# ================================================================ TODO 1 / 2
def training_step(model, x0, y, alpha_bar, T, p_uncond=0.1):
    """배치 하나에 대한 손실을 계산해서 스칼라 텐서로 반환하세요.

    강의 4번 슬라이드의 다섯 줄을 그대로 옮기면 됩니다.

      1. t를 0 이상 T 미만에서 무작위로 뽑습니다.        (배치 크기만큼, long 타입)
      2. eps를 표준정규분포에서 뽑습니다.                (x0와 같은 shape)
      3. x_t = q_sample(x0, t, eps, alpha_bar)
      4. eps_hat = model(x_t, t)        <- Tier 0
         eps_hat = model(x_t, t, y)     <- Tier 1 (조건부)
      5. F.mse_loss(eps_hat, eps) 를 반환

    Tier 1에서 classifier-free guidance를 쓰려면 4번 전에 라벨을 가려야 합니다.
    p_uncond 확률로 y를 model.NULL_CLASS 로 바꾸세요. 세 줄입니다.

        drop = torch.rand(len(y), device=y.device) < p_uncond
        y = torch.where(drop, torch.full_like(y, model.NULL_CLASS), y)

    힌트: t는 x0와 같은 device에 있어야 합니다.
    """
    t = torch.randint(0, T, (len(x0),), device=x0.device, dtype=torch.long)
    eps = torch.randn_like(x0)
    x_t = q_sample(x0, t, eps, alpha_bar)
    if y is None:                                    # Tier 0
        eps_hat = model(x_t, t)
    else:                                            # Tier 1 (CFG label dropout)
        drop = torch.rand(len(y), device=y.device) < p_uncond
        y = torch.where(drop, torch.full_like(y, model.NULL_CLASS), y)
        eps_hat = model(x_t, t, y)
    return F.mse_loss(eps_hat, eps)


# ================================================================ 학습 루프 (그대로 두셔도 됩니다)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1])
    ap.add_argument("--T", type=int, default=None, help="기본값: tier0=200, tier1=1000")
    ap.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--steps", type=int, default=60000, help="tier 0: 총 스텝 수")
    ap.add_argument("--epochs", type=int, default=20, help="tier 1: epoch 수")
    ap.add_argument("--bs", type=int, default=None, help="기본값: tier0=512, tier1=128")
    ap.add_argument("--lr", type=float, default=None, help="기본값: tier0=1e-3, tier1=2e-4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_uncond", type=float, default=0.1, help="tier 1: CFG 라벨 드롭 확률")
    ap.add_argument("--base", type=int, default=96, help="tier 1: SmallUNet 채널 폭")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    T = a.T or (200 if a.tier == 0 else 1000)
    bs = a.bs or (512 if a.tier == 0 else 128)
    lr = a.lr or (1e-3 if a.tier == 0 else 2e-4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tag = f"tier{a.tier}_{a.schedule}_seed{a.seed}"

    print(f"device={device}  tier={a.tier}  T={T}  schedule={a.schedule}  "
          f"bs={bs}  lr={lr}  seed={a.seed}")

    # ---- 데이터
    if a.tier == 0:
        x = D.spiral(3000).to(device)
        y = None
        model = M.TimeMLP().to(device)
    else:
        x, y = D.mnist(train=True)
        x, y = x.to(device), y.to(device)
        model = M.SmallUNet(base=a.base, cond=True).to(device)
    check_data(x, "학습 데이터")
    print(f"데이터 {tuple(x.shape)}  범위 [{x.min():.2f}, {x.max():.2f}]  "
          f"파라미터 {sum(p.numel() for p in model.parameters()):,}")

    _, alpha_bar = make_schedule(T, a.schedule, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    total = a.steps if a.tier == 0 else a.epochs * (len(x) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)

    losses, t0, step = [], time.time(), 0
    n_epochs = 1 if a.tier == 0 else a.epochs
    for ep in range(n_epochs):
        run, nb = 0.0, 0
        perm = torch.randperm(len(x), device=device)
        n_batches = a.steps if a.tier == 0 else len(x) // bs
        for b in range(n_batches):
            if a.tier == 0:
                idx = torch.randint(0, len(x), (bs,), device=device)
            else:
                idx = perm[b * bs:(b + 1) * bs]
            xb = x[idx]
            yb = y[idx] if y is not None else None

            loss = training_step(model, xb, yb, alpha_bar, T, p_uncond=a.p_uncond)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item(); nb += 1; step += 1
            if a.tier == 0 and step % 5000 == 0:
                print(f"  step {step:>6}/{a.steps}  loss {run/nb:.4f}  "
                      f"({time.time()-t0:.0f}s)")

        losses.append(run / max(nb, 1))
        if a.tier == 1:
            print(f"  epoch {ep+1}/{a.epochs}  loss {losses[-1]:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            _save_epoch_grid(model, alpha_bar, T, a, ep, device)

    mins = (time.time() - t0) / 60
    ckpt = os.path.join(a.out, f"{tag}.pt")
    torch.save({"model": model.state_dict(), "T": T, "schedule": a.schedule,
                "tier": a.tier, "seed": a.seed, "base": a.base}, ckpt)
    json.dump({"losses": losses, "train_minutes": mins, "T": T,
               "schedule": a.schedule, "seed": a.seed, "lr": lr, "p_uncond": a.p_uncond,
               "base": a.base, "epochs": a.epochs,
               "n_params": sum(p.numel() for p in model.parameters())},
              open(os.path.join(a.out, f"{tag}_train.json"), "w"), indent=2)
    print(f"\n{mins:.1f}분  ->  {ckpt}")
    print("손실 곡선만 보고 판단하지 마세요. runs/grids 의 샘플 그리드를 보셔야 합니다.")


def _save_epoch_grid(model, alpha_bar, T, a, ep, device):
    """매 epoch 고정 시드 샘플 그리드. 이게 진짜 손실 곡선입니다."""
    try:
        from sample import sample_loop
        import metrics
    except Exception:
        return
    d = os.path.join(a.out, "grids")
    os.makedirs(d, exist_ok=True)
    try:
        model.eval()
        with torch.no_grad():
            y = (torch.arange(64, device=device) % 10)
            s = sample_loop(model, (64, 1, 28, 28), alpha_bar, T, 50,
                            y=y, w=1.0, seed=0, device=device)
        metrics.save_grid(s, os.path.join(d, f"ep{ep+1:02d}.png"))
    except NotImplementedError:
        if ep == 0:
            print("    (sample_loop 이 아직 비어 있어서 그리드 저장을 건너뜁니다)")
    finally:
        model.train()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
