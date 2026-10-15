# -*- coding: utf-8 -*-
"""데이터. Tier 0은 2차원 나선(코드로 생성), Tier 1은 MNIST(torchvision).

Tier 0 나선은 어디서 받아오는 게 아니라 아래 spiral() 이 만들어 냅니다.
반지름이 0~1이라 이미 [-1, 1] 안에 들어가 있고, 그래서 별도 정규화가 없습니다.
"""
import numpy as np
import torch


# ---------------------------------------------------------------- Tier 0
def spiral(n=3000, noise=0.005, seed=0):
    """팔 두 개짜리 나선. 반환값 (n, 2) float32, 대략 [-1, 1] 범위."""
    rng = np.random.default_rng(seed)
    n_per = n // 2
    out = []
    for arm in range(2):
        t = np.sqrt(rng.uniform(0.06, 1.0, n_per)) * 2.4 * np.pi
        r = t / (2.4 * np.pi)                       # 반지름을 0~1로 정규화
        phi = t + arm * np.pi                       # 팔 두 개를 pi만큼 어긋나게
        out.append(np.stack([r * np.cos(phi), r * np.sin(phi)], 1))
    x = np.concatenate(out, 0) + rng.normal(0, noise, (n_per * 2, 2))
    return torch.from_numpy(x.astype(np.float32))


def true_manifold(m=200_000):
    """노이즈 없는 나선을 촘촘히 깐 참조 집합. metrics.distance_to_manifold 용."""
    t = np.linspace(0.06, 1.0, m // 2)
    t = np.sqrt(t) * 2.4 * np.pi
    r = t / (2.4 * np.pi)
    pts = [np.stack([r * np.cos(t + arm * np.pi), r * np.sin(t + arm * np.pi)], 1)
           for arm in range(2)]
    return torch.from_numpy(np.concatenate(pts, 0).astype(np.float32))


# ---------------------------------------------------------------- Tier 1
def mnist(root="./data", train=True):
    """MNIST를 (N, 1, 28, 28) 텐서로. 값은 [-1, 1].

    Normalize((0.5,), (0.5,)) 가 여기 들어 있습니다. 이걸 빼면 [0, 1]이 되고,
    그러면 순방향 과정의 종착점이 N(0, I)가 아니게 됩니다.
    """
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.MNIST(root, train=train, download=True, transform=tf)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    return x, y


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    s = spiral()
    print(f"spiral  {tuple(s.shape)}  범위 [{s.min():.2f}, {s.max():.2f}]")
    print(f"manifold {tuple(true_manifold(20000).shape)}")
    x, y = mnist()
    print(f"mnist   {tuple(x.shape)}  범위 [{x.min():.2f}, {x.max():.2f}]  "
          f"라벨 {y.min()}~{y.max()}")
