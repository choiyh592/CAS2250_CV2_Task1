# -*- coding: utf-8 -*-
"""디노이저 두 개. Tier 0은 3층 MLP, Tier 1은 작은 U-Net.

둘 다 하는 일은 똑같습니다. (노이즈 낀 입력, 타임스텝) 을 받아서 노이즈를 예측합니다.
디퓨전이 U-Net을 요구하는 게 아니라 이미지가 요구하는 거예요. 2차원에서는 MLP로 충분합니다.

diffusers 의 UNet2DModel 로 갈아 끼우셔도 됩니다. 단 학습된 가중치가 아니라
구조만 가져오셔야 하고, forward 시그니처만 맞춰 주면 나머지 코드는 그대로 돕니다.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """t 하나를 dim 차원 벡터로. 이게 없으면 모델이 모든 노이즈 레벨을 평균냅니다."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], -1)


# ---------------------------------------------------------------- Tier 0
class TimeMLP(nn.Module):
    """2차원용 3층 MLP + time embedding. hidden=512 에서 파라미터 823,810개.

    U-Net과 같은 방식으로 time embedding을 **모든 층에** 더합니다. 입력에서 한 번만
    이어 붙이면 t 정보가 깊은 층까지 전달되지 않아 눈에 띄게 나빠집니다.
    (직접 재 보면 200스텝 d가 0.0034 대신 0.0077 정도로 두 배가 됩니다.)
    """

    def __init__(self, hidden=512, tdim=64):
        super().__init__()
        self.tdim = tdim
        self.tproj = nn.Sequential(nn.Linear(tdim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden))
        self.l1 = nn.Linear(2, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 2)
        self.act = nn.SiLU()

    def forward(self, x, t):
        te = self.tproj(timestep_embedding(t, self.tdim))
        h = self.act(self.l1(x) + te)      # <- 매 층에 te를 더합니다
        h = self.act(self.l2(h) + te)
        h = self.act(self.l3(h) + te)
        return self.out(h)


# ---------------------------------------------------------------- Tier 1
class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.temb(F.silu(t))[:, :, None, None]   # time embedding이 들어가는 자리
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28. base=96 이면 파라미터 약 350만 개.

    cond=True 면 라벨 임베딩이 붙습니다. 마지막 인덱스(n_classes)가 classifier-free
    guidance 에서 쓰는 '라벨 없음' 토큰입니다. NULL_CLASS 로 꺼내 쓰세요.
    """

    def __init__(self, base=96, tdim=256, n_classes=10, cond=True):
        super().__init__()
        self.cond, self.tdim, self.n_classes = cond, tdim, n_classes
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        if cond:
            self.cemb = nn.Embedding(n_classes + 1, tdim)
        c1, c2, c3 = base, base * 2, base * 2
        self.inp = nn.Conv2d(1, c1, 3, padding=1)
        self.d1 = ResBlock(c1, c1, tdim)
        self.ds1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2 = ResBlock(c1, c2, tdim)
        self.ds2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.m1 = ResBlock(c2, c3, tdim)
        self.m2 = ResBlock(c3, c2, tdim)
        self.u2 = ResBlock(c2 + c2, c1, tdim)
        self.u1 = ResBlock(c1 + c1, c1, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(),
                                 nn.Conv2d(c1, 1, 3, padding=1))

    @property
    def NULL_CLASS(self):
        """'라벨 없음' 토큰. 학습 중 라벨을 가릴 때와 무조건부 예측에 씁니다."""
        return self.n_classes

    def forward(self, x, t, y=None):
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        if self.cond:
            if y is None:
                y = torch.full((x.shape[0],), self.NULL_CLASS,
                               device=x.device, dtype=torch.long)
            temb = temb + self.cemb(y)
        h1 = self.d1(self.inp(x), temb)
        h2 = self.d2(self.ds1(h1), temb)
        h3 = self.m2(self.m1(self.ds2(h2), temb), temb)
        u = F.interpolate(h3, scale_factor=2, mode="nearest")
        u = self.u2(torch.cat([u, h2], 1), temb)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.u1(torch.cat([u, h1], 1), temb)
        return self.out(u)


def build(tier, **kw):
    return TimeMLP(**kw) if tier == 0 else SmallUNet(**kw)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    m0 = TimeMLP()
    m1 = SmallUNet(base=96)
    print(f"TimeMLP    {sum(p.numel() for p in m0.parameters()):>10,} 파라미터")
    print(f"SmallUNet  {sum(p.numel() for p in m1.parameters()):>10,} 파라미터  (강의 실측 3,507,553)")
    t = torch.randint(0, 1000, (4,))
    print("shape 확인:", m0(torch.randn(4, 2), t).shape,
          m1(torch.randn(4, 1, 28, 28), t, torch.tensor([1, 2, 3, 4])).shape)
