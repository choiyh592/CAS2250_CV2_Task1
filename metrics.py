# -*- coding: utf-8 -*-
"""채점 대상인 네 축을 재는 코드. 강의 슬라이드의 계산법 상자를 그대로 구현한 것입니다.

  fidelity        개별 샘플이 진짜 데이터답게 생겼는가
  diversity       coverage(클래스 히스토그램) + variety(같은 클래스 안의 다양성)
  memorization    학습셋을 그대로 베낀 건 아닌가
  cost            그 품질에 뭘 지불했는가  (시간은 여러분 코드에서 재세요)

이걸 통일해서 드리는 이유는, 팀마다 다른 방식으로 재면 숫자를 비교할 수 없기 때문입니다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ================================================================ Tier 0: d
@torch.no_grad()
def distance_to_manifold(samples, manifold, device="cpu", chunk=512):
    """각 샘플에서 참 곡선까지의 최근접 거리 평균. 슬라이드의 d 입니다.

    manifold 는 data.true_manifold() 로 만든 노이즈 없는 참조 점들입니다.
    정답 곡선을 우리가 알고 있어서 2차원에서만 가능한 지표예요.
    """
    s = samples.to(device)
    m = manifold.to(device)
    tot = 0.0
    for i in range(0, len(s), chunk):
        tot += torch.cdist(s[i:i + chunk], m).min(dim=1).values.sum().item()
    return tot / len(s)


# ================================================================ Tier 1: 평가용 분류기
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def train_classifier(xtr, ytr, xte, yte, device="cpu", epochs=3, min_acc=0.98):
    """진짜 MNIST로 평가용 분류기를 학습합니다. 3 epoch면 98%대가 나옵니다.

    이 분류기가 test set에서 min_acc 미만이면 그걸로 잰 fidelity는 의미가 없으므로
    여기서 막습니다. 과제 규정과 같은 값(98%)입니다.
    """
    clf = SmallCNN().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    dl = DataLoader(TensorDataset(xtr, ytr), batch_size=256, shuffle=True)
    for _ in range(epochs):
        clf.train()
        for xb, yb in dl:
            loss = F.cross_entropy(clf(xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        ok = sum((clf(xte[i:i + 1000].to(device)).argmax(-1).cpu()
                  == yte[i:i + 1000]).sum().item() for i in range(0, len(xte), 1000))
    acc = ok / len(xte)
    assert acc >= min_acc, (
        f"평가용 분류기 정확도가 {acc:.2%}입니다. {min_acc:.0%} 이상이어야 "
        f"이 분류기로 잰 fidelity를 신뢰할 수 있습니다. epoch를 늘려 보세요.")
    return clf, acc


@torch.no_grad()
def _probs(clf, x, device, chunk=1000):
    out = [F.softmax(clf(x[i:i + chunk].to(device)), -1).cpu()
           for i in range(0, len(x), chunk)]
    return torch.cat(out)


def fidelity(clf, samples, device="cpu", thresh=0.9):
    """분류기가 확신(p > thresh)을 갖고 라벨을 붙이는 샘플의 비율."""
    conf, _ = _probs(clf, samples, device).max(-1)
    return float((conf >= thresh).float().mean())


def coverage(clf, samples, device="cpu", n_classes=10):
    """예측 클래스 히스토그램. 평평한 게 목표입니다. (히스토그램, 엔트로피) 반환.

    주의: 클래스를 균등하게 요청하고 모델이 따르면 엔트로피는 구조적으로 최대입니다.
    그래서 이것만으로는 다양성을 잰 게 아닙니다. within_class_variety 를 같이 보세요.
    """
    pred = _probs(clf, samples, device).argmax(-1)
    h = torch.bincount(pred, minlength=n_classes).float()
    h = h / h.sum()
    ent = float(-(h * (h + 1e-12).log()).sum())
    return h.tolist(), ent


def class_match(clf, samples, wanted, device="cpu"):
    """요청한 클래스가 실제로 나왔는지의 비율."""
    pred = _probs(clf, samples, device).argmax(-1)
    return float((pred == wanted).float().mean())


# ================================================================ diversity: variety
@torch.no_grad()
def mean_pairwise(x, device="cpu", cap=400):
    """한 묶음 안의 평균 쌍거리(L2). 같은 클래스끼리 넣으세요."""
    v = x[:cap].reshape(min(len(x), cap), -1).to(device)
    n = len(v)
    return float(torch.cdist(v, v).sum().item() / (n * (n - 1)))


def within_class_variety(samples_one_class, real_one_class, device="cpu"):
    """같은 클래스 샘플의 다양성을, 진짜 데이터의 같은 값 대비 비율로.

    슬라이드의 '실제 데이터 대비 90%'가 이 값입니다. guidance를 키우면 여기가
    떨어집니다. coverage는 전혀 움직이지 않고요.
    """
    gen = mean_pairwise(samples_one_class, device)
    real = mean_pairwise(real_one_class, device)
    return gen / real, gen, real


# ================================================================ memorization
@torch.no_grad()
def nn_to_train(samples, train_set, device="cpu", chunk=200):
    """생성 샘플에서 학습셋까지의 L2 최근접 거리 평균.

    이 값 하나만으로는 아무 의미가 없습니다. 반드시 진짜 held-out 데이터로 잰
    같은 값과 비교하세요. 생성 쪽이 더 가까우면 베끼고 있다는 뜻입니다.
    """
    s = samples.reshape(len(samples), -1).to(device)
    r = train_set.reshape(len(train_set), -1).to(device)
    tot = 0.0
    for i in range(0, len(s), chunk):
        tot += torch.cdist(s[i:i + chunk], r).min(dim=1).values.sum().item()
    return tot / len(s)


# ================================================================ 그림
def save_grid(samples, path, nrow=8):
    """큐레이션하지 않은 nrow x nrow 그리드. 보고서 필수 그림입니다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = nrow * nrow
    x = ((samples[:n] + 1) / 2).clamp(0, 1).squeeze(1).cpu().numpy()
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow, nrow))
    for ax, img in zip(axes.flat, x):
        ax.imshow(img, cmap="gray"); ax.axis("off")
    fig.tight_layout(pad=0.1); fig.savefig(path, dpi=110); plt.close(fig)
    return path
