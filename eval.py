# -*- coding: utf-8 -*-
"""네 축을 전부 재서 results.json 으로 떨굽니다. 보고서 숫자는 여기서 나옵니다.

    python eval.py --ckpt runs/tier0_cosine_seed0.pt --steps 3 5 10 20 50 200
    python eval.py --ckpt runs/tier1_cosine_seed0.pt --steps 50 --w 0 1 2 3 5 8

TODO가 없습니다. train.py 와 sample.py 를 채우고 나면 그대로 돕니다.
"""
import argparse
import json
import os
import time

import torch

import data as D
import metrics
from diffusion import make_schedule
from sample import load, sample_loop


def eval_tier0(model, alpha_bar, c, a, device):
    man = D.true_manifold().to(device)
    real = D.spiral(2000, seed=99).to(device)            # held-out 실데이터
    floor = metrics.distance_to_manifold(real, man, device)
    out = {"floor_real_heldout": floor, "steps": {}}
    print(f"[기준] 진짜 held-out 데이터의 d = {floor:.4f}  (이 아래로는 못 내려갑니다)\n")
    print(f"{'steps':>6} {'d':>10} {'sec/100':>10}")
    for ns in a.steps:
        t0 = time.time()
        with torch.no_grad():
            x = sample_loop(model, (a.n, 2), alpha_bar, c["T"], ns,
                            seed=a.seed, device=device)
        dt = time.time() - t0
        d = metrics.distance_to_manifold(x, man, device)
        out["steps"][ns] = {"d": d, "sec_per_100": dt / (a.n / 100)}
        print(f"{ns:>6} {d:>10.4f} {dt/(a.n/100):>10.2f}")
    return out


def eval_tier1(model, alpha_bar, c, a, device):
    xtr, ytr = D.mnist(train=True)
    xte, yte = D.mnist(train=False)
    print("평가용 분류기를 학습합니다 (3 epoch)...")
    clf, acc = metrics.train_classifier(xtr, ytr, xte, yte, device)
    print(f"  분류기 test 정확도 {acc:.2%}   (98% 이상이어야 fidelity가 유효합니다)\n")

    n = a.n
    labels = (torch.arange(n) % 10)
    real_nn = metrics.nn_to_train(xte[:n], xtr, device)
    out = {"classifier_acc": acc, "real": {"nn_to_train": real_nn,
                                           "fidelity": metrics.fidelity(clf, xte[:n], device)},
           "w": {}}
    print(f"[기준] 진짜 held-out 데이터: NN-to-train {real_nn:.2f}\n")
    print(f"{'w':>5} {'fidelity':>9} {'match':>7} {'coverage H':>11} "
          f"{'variety':>9} {'NN-train':>9}")

    digit = 7
    real7 = xte[yte == digit][:300]
    for w in a.w:
        with torch.no_grad():
            x = sample_loop(model, (n, 1, 28, 28), alpha_bar, c["T"], a.steps[0],
                            y=labels.to(device), w=w, seed=a.seed, device=device)
            s7 = sample_loop(model, (300, 1, 28, 28), alpha_bar, c["T"], a.steps[0],
                             y=torch.full((300,), digit, device=device),
                             w=w, seed=7, device=device)
        fid = metrics.fidelity(clf, x, device)
        hist, ent = metrics.coverage(clf, x, device)
        match = metrics.class_match(clf, x, labels, device)
        ratio, gen_pw, real_pw = metrics.within_class_variety(s7, real7, device)
        nnd = metrics.nn_to_train(x, xtr, device)
        out["w"][w] = {"fidelity": fid, "class_match": match, "coverage_entropy": ent,
                       "coverage_hist": hist, "variety_ratio": ratio,
                       "nn_to_train": nnd}
        print(f"{w:>5.1f} {fid:>9.3f} {match:>7.3f} {ent:>11.3f} "
              f"{ratio:>8.0%} {nnd:>9.2f}")
        metrics.save_grid(x, os.path.join(a.out, f"grid_w{w:g}.png"))

    print("\ncoverage 엔트로피가 w를 바꿔도 안 움직이는 걸 확인하세요. "
          "그게 슬라이드에서 말한 '움직이지 않는 지표'입니다.")
    print("variety 는 떨어집니다. 그게 guidance의 진짜 대가입니다.")
    if min(v["nn_to_train"] for v in out["w"].values()) < real_nn:
        print("\n경고: 생성 샘플이 진짜 데이터보다 학습셋에 더 가깝습니다 -> 암기 의심")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[50])
    ap.add_argument("--w", type=float, nargs="+", default=[0.0, 1.0, 3.0])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    model, c = load(a.ckpt, device)
    _, alpha_bar = make_schedule(c["T"], c["schedule"], device)
    print(f"device={device}  tier={c['tier']}  T={c['T']}  schedule={c['schedule']}\n")

    res = (eval_tier0 if c["tier"] == 0 else eval_tier1)(model, alpha_bar, c, a, device)
    res["ckpt"] = os.path.basename(a.ckpt)
    res["tier"] = c["tier"]
    path = os.path.join(a.out, f"results_tier{c['tier']}_seed{a.seed}.json")
    json.dump(res, open(path, "w"), indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
