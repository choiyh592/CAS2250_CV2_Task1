# -*- coding: utf-8 -*-
"""디퓨전의 뼈대: 노이즈 스케줄, 순방향 과정, DDIM 한 스텝.

여러분이 고칠 파일이 아닙니다. 그냥 import 해서 쓰세요.
학습 루프(train.py)와 샘플링 루프(sample.py)가 여러분이 작성할 부분입니다.

여기에 들어 있는 세 가지 안전장치는 강의에서 말씀드린 지뢰 세 개에 대응합니다.

  1. 정규화 누락      check_data() 가 [0, 1] 데이터를 잡습니다.
  2. alpha_bar 발산   make_schedule() 이 하한을 걸고, 끝값이 0인지 assert 합니다.
  3. DDIM 자기모순    ddim_step() 이 클램프한 x0에서 eps를 다시 유도합니다.

3번은 증상이 고약합니다. 스텝을 늘릴수록 품질이 나빠져요. 실제로 재 보면 샘플
표준편차가 10스텝 0.68에서 50스텝 0.305로 떨어집니다. 재유도하면 0.98로 유지되고요.
"""
import math

import torch

AB_FLOOR = 1e-5  # alpha_bar 하한. 0으로 두면 1/sqrt(alpha_bar)가 발산합니다.


# ---------------------------------------------------------------- 스케줄
def make_schedule(T, kind="cosine", device="cpu"):
    """betas, alpha_bar를 만들어 돌려줍니다. alpha_bar는 하한으로 클램프됩니다."""
    if kind == "linear":
        # DDPM 원논문 값은 T=1000 기준입니다. T를 바꾸면 같이 스케일해야
        # alpha_bar가 여전히 0까지 내려갑니다.
        scale = 1000.0 / T
        betas = torch.linspace(scale * 1e-4, scale * 0.02, T, dtype=torch.float64)
    elif kind == "cosine":
        s = 0.008
        ts = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
        ab = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
        betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    else:
        raise ValueError(f"모르는 스케줄입니다: {kind!r}  (linear 또는 cosine)")

    betas = betas.float().to(device)
    alpha_bar = torch.cumprod(1 - betas, 0).clamp(min=AB_FLOOR)

    assert alpha_bar[-1] <= 1e-4, (
        f"alpha_bar가 t=T에서 {alpha_bar[-1]:.4f}로 끝납니다. 0에 도달해야 합니다. "
        f"linear라면 beta를 T에 맞게 스케일했는지 확인하세요.")
    return betas, alpha_bar


# ---------------------------------------------------------------- 데이터 점검
def check_data(x, name="batch"):
    """데이터가 [-1, 1] 범위인지 확인합니다. 학습 시작 전에 한 번 부르세요."""
    lo, hi = float(x.min()), float(x.max())
    assert -1.05 <= lo and hi <= 1.05, (
        f"{name}의 범위가 [{lo:.2f}, {hi:.2f}]입니다. [-1, 1]을 벗어났습니다.")
    # ToTensor만 쓰면 [0, 1]이 나오는데, 그 범위도 [-1, 1] 안에 들어가므로
    # 위 검사만으로는 통과해 버립니다. 이게 진짜 잡고 싶은 경우입니다.
    assert lo < -0.5, (
        f"{name}의 범위가 [{lo:.2f}, {hi:.2f}]입니다. [0, 1] 데이터로 보입니다. "
        f"transforms.Normalize((0.5,), (0.5,))를 빠뜨렸습니다.")
    assert hi > 0.5, (
        f"{name}의 최댓값이 {hi:.2f}로 너무 작습니다. 이미 정규화된 데이터를 "
        f"한 번 더 정규화한 것은 아닌지 확인하세요.")
    return x


# ---------------------------------------------------------------- 순방향
def q_sample(x0, t, noise, alpha_bar):
    """x_t = sqrt(ab_t) * x0 + sqrt(1 - ab_t) * noise  —  강의의 그 한 줄입니다.

    x0 : (B, ...)      원본
    t  : (B,) long     각 샘플의 타임스텝
    """
    a = alpha_bar[t]
    while a.dim() < x0.dim():          # (B,) -> (B, 1, 1, 1) 등으로 브로드캐스트
        a = a[..., None]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


# ---------------------------------------------------------------- 역방향 한 스텝
def ddim_step(x, eps, t, s, alpha_bar):
    """x_t 에서 x_s 로 내려가는 DDIM 한 스텝. 마지막 스텝이면 s=None.

    이 함수는 여러분에게 그냥 드립니다. 수치적으로 위험한 부분이라 모두가 같은
    조건에서 측정하도록 고정해 두는 게 낫다고 판단했습니다. 샘플링 '루프'와
    guidance 혼합은 sample.py 에서 여러분이 작성하셔야 합니다.
    """
    a_t = alpha_bar[t]
    x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
    if s is None:
        return x0
    # 클램프한 x0에서 eps를 다시 유도합니다. 이 줄을 지우면 x0와 eps가 서로 다른
    # 이야기를 하게 되고, 그 모순이 스텝마다 누적됩니다.
    eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
    a_s = alpha_bar[s]
    return a_s.sqrt() * x0 + (1 - a_s).sqrt() * eps


def timestep_schedule(T, n_steps):
    """T-1 에서 0 까지 n_steps 개로 균등하게 나눈 타임스텝 목록."""
    return torch.linspace(T - 1, 0, n_steps).round().long().tolist()


# ---------------------------------------------------------------- 전체 점검
def sanity_check(T=1000, kind="cosine", device="cpu", verbose=True):
    """학습을 시작하기 전에 한 번 돌려 보세요. 몇 초면 끝납니다."""
    _, ab = make_schedule(T, kind, device)
    if verbose:
        print(f"[schedule] {kind}, T={T}")
        print(f"           alpha_bar[0]   = {ab[0]:.4f}   (1에 가까워야 함)")
        print(f"           alpha_bar[T-1] = {ab[-1]:.6f} (0에 도달해야 함)")

    x0 = torch.randn(256, 1, 28, 28, device=device).clamp(-1, 1)
    check_data(x0, "x0")
    e = torch.randn_like(x0)
    t = torch.full((len(x0),), T - 1, device=device, dtype=torch.long)
    xT = q_sample(x0, t, e, ab)
    m, sd = float(xT.mean()), float(xT.std())
    assert abs(m) < 0.05 and abs(sd - 1.0) < 0.05, (
        f"x_T의 평균/표준편차가 ({m:.3f}, {sd:.3f})입니다. (0, 1)이어야 합니다.")
    if verbose:
        print(f"[forward]  x_T ~ ({m:+.3f}, {sd:.3f})   -> 순수 가우시안 OK")

    for n_steps in (3, 10, 50):
        steps = timestep_schedule(T, n_steps)
        x = torch.randn(16, 1, 28, 28, device=device)
        for i, tt in enumerate(steps):
            s = steps[i + 1] if i + 1 < len(steps) else None
            x = ddim_step(x, torch.randn_like(x), tt, s, ab)
        assert torch.isfinite(x).all(), (
            f"{n_steps}스텝 샘플링에서 NaN이 나왔습니다. alpha_bar 클램프를 확인하세요.")
        if verbose:
            print(f"[sampler]  {n_steps:>3}스텝  NaN 없음, 범위 "
                  f"[{float(x.min()):+.2f}, {float(x.max()):+.2f}]")

    if verbose:
        print("\n전부 통과했습니다. 이제 학습을 시작하세요.")
    return True


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for k in ("cosine", "linear"):
        sanity_check(T=1000, kind=k)
        print()
