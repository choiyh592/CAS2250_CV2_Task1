# -*- coding: utf-8 -*-
"""샘플링. 여기 TODO 하나가 여러분이 채울 두 번째 부분입니다.

    python sample.py --ckpt runs/tier1_cosine_seed0.pt --steps 50 --w 1.0
    python sample.py --ckpt runs/tier0_cosine_seed0.pt --steps 200 --n 3000

한 스텝을 내려가는 ddim_step 은 diffusion.py 에 이미 있습니다. 수치적으로
위험한 부분이라 모두가 같은 조건에서 측정하도록 고정해 뒀습니다.
여러분이 쓰실 것은 그 스텝을 엮는 '루프'와 guidance 혼합입니다.
"""
import argparse
import os

import torch

import models as M
from diffusion import make_schedule, ddim_step, timestep_schedule


# ================================================================ TODO 2 / 2
def sample_loop(model, shape, alpha_bar, T, n_steps, y=None, w=0.0,
                seed=0, device="cpu"):
    """노이즈에서 시작해 샘플을 만들어 반환하세요. 반환 shape은 인자 shape 그대로.

    강의 5번 슬라이드의 의사코드입니다.

      1. g = torch.Generator(device=device).manual_seed(seed)
         x = torch.randn(shape, device=device, generator=g)
      2. steps = timestep_schedule(T, n_steps)          # T-1 -> 0
      3. steps를 순서대로 돌면서:
           tt = torch.full((len(x),), t, device=device, dtype=torch.long)
           eps = <아래 guidance 설명 참고>
           s   = 다음 스텝 (마지막이면 None)
           x   = ddim_step(x, eps, t, s, alpha_bar)
      4. x 반환

    guidance (Tier 1, w > 0일 때):
        조건부 예측과 무조건부 예측을 각각 구해서 섞습니다.
            ec  = model(x, tt, y)
            eu  = model(x, tt, y가 전부 model.NULL_CLASS 인 텐서)
            eps = ec + w * (ec - eu)
        w == 0 이면 그냥 eps = model(x, tt, y) 로 두세요. Tier 0은 라벨이 없으니
        eps = model(x, tt) 입니다.

    주의: 반드시 torch.no_grad() 안에서 도세요. 안 그러면 메모리가 터집니다.
    """
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, device=device, generator=g)
        steps = timestep_schedule(T, n_steps)
        for i, t in enumerate(steps):
            tt = torch.full((len(x),), t, device=device, dtype=torch.long)
            if y is None:                            # Tier 0
                eps = model(x, tt)
            elif w == 0:
                eps = model(x, tt, y)
            else:                                    # classifier-free guidance
                ec = model(x, tt, y)
                eu = model(x, tt, torch.full_like(y, model.NULL_CLASS))
                eps = ec + w * (ec - eu)
            s = steps[i + 1] if i + 1 < len(steps) else None
            x = ddim_step(x, eps, t, s, alpha_bar)
        return x


# ================================================================ CLI (그대로 두셔도 됩니다)
def load(ckpt, device):
    c = torch.load(ckpt, map_location=device)
    model = (M.TimeMLP() if c["tier"] == 0 else M.SmallUNet(base=c.get("base", 96), cond=True)).to(device)
    model.load_state_dict(c["model"])
    model.eval()
    return model, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/samples")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, c = load(a.ckpt, device)
    _, alpha_bar = make_schedule(c["T"], c["schedule"], device)
    os.makedirs(a.out, exist_ok=True)

    if c["tier"] == 0:
        shape, y = (a.n, 2), None
    else:
        shape = (a.n, 1, 28, 28)
        y = (torch.arange(a.n, device=device) % 10)

    import time
    t0 = time.time()
    with torch.no_grad():
        x = sample_loop(model, shape, alpha_bar, c["T"], a.steps,
                        y=y, w=a.w, seed=a.seed, device=device)
    dt = time.time() - t0
    assert torch.isfinite(x).all(), (
        "샘플에 NaN이 있습니다. alpha_bar 클램프와 x0 클리핑을 확인하세요.")

    base = os.path.join(a.out, f"s{a.steps}_w{a.w:g}_seed{a.seed}")
    torch.save(x.cpu(), base + ".pt")
    if c["tier"] == 1:
        import metrics
        metrics.save_grid(x, base + ".png")
    print(f"{a.n}개 샘플, {a.steps}스텝, w={a.w}  ->  {base}.pt")
    print(f"100샘플당 {dt / (a.n / 100):.2f}초   (cost 축에 보고할 숫자입니다)")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
