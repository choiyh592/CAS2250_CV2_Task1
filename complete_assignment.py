# -*- coding: utf-8 -*-
"""과제 전체를 한 번에: 학습 -> 평가 -> results/ 정리 -> summary.md -> 보고서 그림.

    Tier 0  나선   cosine / linear  x  시드 0,1,2                     (lr 1e-3, train.py 기본값)
    Tier 1  MNIST  main: cosine, lr 1e-3, p_uncond 0.1, 20 epoch, base 96  x  시드 0,1,2
            ablation (main 에서 한 가지만 바꿈, 시드 3개씩):
              lr 2e-4, p_uncond 0.2, 5/10 epoch, base 64/128

인자 없이 실행합니다. 결과물이 이미 있는 단계는 건너뛰므로, 중간에 끊기면 그냥 다시 돌리세요.
실패한 단계는 logs/<단계>.failed.txt 로 남기고 다음 단계로 넘어갑니다.
"""
import json
import shutil
import statistics as st
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- 설정
SEEDS = [0, 1, 2]
T0_SCHEDULES = ["cosine", "linear"]
T0_TRAIN_STEPS = 60000
T0_EVAL_STEPS = [3, 5, 10, 20, 50, 200]
T1_EPOCHS = 20
T1_LR = 1e-3                      # 슬라이드 15 의 기준 설정 (train.py 의 tier 1 기본값은 2e-4)
T1_SAMPLE_STEPS = 50
T1_W = [0, 1, 2, 3, 5, 8]         # main 의 guidance sweep
ABL_W = [0, 1, 3]                 # ablation 은 평가 시간을 줄이려고 세 점만
ABLATIONS = {                     # 이름: main 대비 바꾸는 train.py 인자. 이 순서대로 돕니다 (싼 것 먼저, base128 맨 끝)
    "lr2e-4": ["--lr", 2e-4],     # train.py 의 tier 1 기본값. 결과는 OLD_LR2E4 경로에 저장
    "p_uncond0.2": ["--p_uncond", 0.2],
    "ep5": ["--epochs", 5],
    "ep10": ["--epochs", 10],
    "base64": ["--base", 64],
    "base128": ["--base", 128],
}
OLD_LR2E4 = {0: Path("runs"), 1: Path("runs/seed1"), 2: Path("runs/seed2")}   # lr 2e-4 로 이미 끝난 run
LOGS = Path("logs")
RESULTS = Path("results")
FAILED = []


def run(name, done, *args):
    """done 경로가 있으면 건너뜀. 실패하면 로그를 *.failed.txt 로 남기고 False."""
    if Path(done).exists():
        print(f"[skip] {name}", flush=True)
        return True
    print(f"[run ] {name}", flush=True)
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"{name}.txt"
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, *map(str, args)], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        log.replace(LOGS / f"{name}.failed.txt")
        FAILED.append(name)
        print(f"[FAIL] {name} -> logs/{name}.failed.txt", flush=True)
        return False
    return True


def t1_dir(cfg, s):
    if cfg == "lr2e-4":
        return OLD_LR2E4[s]
    return Path("runs") / ("main" if cfg == "main" else f"abl/{cfg}") / f"seed{s}"


def t1_train_args(cfg, s):
    """main 설정에 ablation 인자를 덮어쓴 train.py 인자 목록 (같은 플래그를 두 번 넘기지 않음)."""
    opts = {"--epochs": T1_EPOCHS, "--schedule": "cosine", "--lr": T1_LR}
    extra = [] if cfg == "main" else ABLATIONS[cfg]
    opts.update(zip(extra[::2], extra[1::2]))
    return ["train.py", "--tier", 1, *[x for kv in opts.items() for x in kv], "--seed", s, "--out", t1_dir(cfg, s)]


def tier1(cfg, s, ws, sample=False):
    out = t1_dir(cfg, s)
    ck = out / f"tier1_cosine_seed{s}.pt"
    if not run(f"train_t1_{cfg}_s{s}", ck, *t1_train_args(cfg, s)):
        return
    if sample:
        run(f"sample_t1_{cfg}_s{s}", LOGS / f"sample_t1_{cfg}_s{s}.txt", "sample.py", "--ckpt", ck,
            "--steps", T1_SAMPLE_STEPS, "--w", 1.0, "--seed", s, "--out", out / "samples")
    run(f"eval_t1_{cfg}_s{s}", out / "eval" / f"results_tier1_seed{s}.json", "eval.py", "--ckpt", ck,
        "--steps", T1_SAMPLE_STEPS, "--w", *ws, "--seed", s,
        "--out", out / "eval")


def tier0():
    for sch in T0_SCHEDULES:
        for s in SEEDS:
            ck = Path("runs") / f"tier0_{sch}_seed{s}.pt"
            ev = Path("runs") / f"eval_tier0_{sch}"      # 스케줄별로 나눔 (eval.py 파일명이 시드만 담아서 겹침)
            if run(f"train_t0_{sch}_s{s}", ck, "train.py", "--tier", 0, "--steps", T0_TRAIN_STEPS,
                   "--schedule", sch, "--seed", s):
                run(f"eval_t0_{sch}_s{s}", ev / f"results_tier0_seed{s}.json", "eval.py", "--ckpt", ck,
                    "--steps", *T0_EVAL_STEPS, "--seed", s, "--out", ev)


# ---------------------------------------------------------------- results/ 정리
def cp(src, dst):
    src, dst = Path(src), RESULTS / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.unlink(missing_ok=True)       # 예전 run 의 낡은 파일이 main 인 척 남지 않도록


def collect():
    """README '제출물' 구조. results/ 의 tier1 숫자는 main (lr 1e-3), ablation 은 results/ablations/."""
    m0 = t1_dir("main", 0)
    for p in sorted((m0 / "grids").glob("ep*.png")):
        cp(p, Path("grids") / p.name)
    cp(m0 / "samples" / f"s{T1_SAMPLE_STEPS}_w1_seed0.png", "grid_8x8.png")
    for sch in T0_SCHEDULES:
        for s in SEEDS:
            cp(Path("runs") / f"eval_tier0_{sch}" / f"results_tier0_seed{s}.json", f"results_tier0_{sch}_seed{s}.json")
            cp(Path("runs") / f"tier0_{sch}_seed{s}_train.json", f"tier0_{sch}_seed{s}_train.json")
    for s in SEEDS:
        cp(t1_dir("main", s) / "eval" / f"results_tier1_seed{s}.json", f"results_tier1_seed{s}.json")
        cp(t1_dir("main", s) / f"tier1_cosine_seed{s}_train.json", f"tier1_cosine_seed{s}_train.json")
    for cfg in ABLATIONS:
        for s in SEEDS:
            d = t1_dir(cfg, s)
            cp(d / "eval" / f"results_tier1_seed{s}.json", Path("ablations") / f"{cfg}_seed{s}.json")
            cp(d / f"tier1_cosine_seed{s}_train.json", Path("ablations") / f"{cfg}_seed{s}_train.json")


# ---------------------------------------------------------------- summary.md
def ms(xs, fmt=".4f"):
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:{fmt}} (n=1)"
    return f"{st.mean(xs):{fmt}} ± {st.stdev(xs):{fmt}}"


def loads(names):
    return [json.load(open(RESULTS / n, encoding="utf-8")) for n in names if (RESULTS / n).exists()]


def summarize():
    L = [f"# 결과 요약 (시드 {SEEDS}, 평균 ± 표준편차)", ""]

    L += ["## Tier 0 — 스텝 수 vs d (낮을수록 좋음)", ""]
    t0 = {sch: loads([f"results_tier0_{sch}_seed{s}.json" for s in SEEDS]) for sch in T0_SCHEDULES}
    L += ["| steps | " + " | ".join(f"d ({sch})" for sch in T0_SCHEDULES) + " | sec/100 (cosine) |",
          "|---|" + "---|" * (len(T0_SCHEDULES) + 1)]
    for ns in T0_EVAL_STEPS:
        ds = [ms([r["steps"][str(ns)]["d"] for r in t0[sch]]) for sch in T0_SCHEDULES]
        L.append(f"| {ns} | " + " | ".join(ds) + f" | {ms([r['steps'][str(ns)]['sec_per_100'] for r in t0['cosine']], '.4f')} |")
    L += ["", f"held-out 실데이터 d (바닥): {ms([r['floor_real_heldout'] for r in t0['cosine']])}",
          "sec/100: eval.py 가 준비 실행·CUDA 동기화 없이 첫 스텝 수(3)부터 재므로, 3스텝 값에는 첫 호출 오버헤드가 섞였을 수 있음", ""]

    L += [f"## Tier 1 main (lr {T1_LR:g}) — guidance w 별 네 축", ""]
    t1 = loads([f"results_tier1_seed{s}.json" for s in SEEDS])
    L += ["| w | fidelity | class match | coverage H | variety | NN-to-train |", "|---|---|---|---|---|---|"]
    for w in T1_W:
        g = lambda k: [r["w"][str(float(w))][k] for r in t1]
        L.append(f"| {w} | {ms(g('fidelity'), '.3f')} | {ms(g('class_match'), '.3f')} | "
                 f"{ms(g('coverage_entropy'), '.3f')} | {ms(g('variety_ratio'), '.3f')} | {ms(g('nn_to_train'), '.2f')} |")
    L.append(f"| real (test, 첫 1,000장) | {ms([r['real']['fidelity'] for r in t1], '.3f')} | — | — | 1 (정의상) | "
             f"{ms([r['real']['nn_to_train'] for r in t1], '.2f')} |")
    L += ["", f"held-out 실데이터 NN-to-train (암기 기준선): {ms([r['real']['nn_to_train'] for r in t1], '.2f')}",
          f"평가용 분류기 정확도: {ms([r['classifier_acc'] for r in t1])}", ""]

    L += ["## Tier 1 ablation — main 에서 한 가지만 바꿈 (w = 0 / 1 / 3)", "",
          "| 설정 | 시드 | params | 학습 분 | fid w0 | fid w1 | match w1 | variety w1 | variety w3 | NN w1 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for cfg in ["main", *ABLATIONS]:
        if cfg == "main":
            ev, tr = t1, loads([f"tier1_cosine_seed{s}_train.json" for s in SEEDS])
        else:
            ev = loads([f"ablations/{cfg}_seed{s}.json" for s in SEEDS])
            tr = loads([f"ablations/{cfg}_seed{s}_train.json" for s in SEEDS])
        if not ev:
            L.append(f"| {cfg} | 0 | | | (아직 없음) | | | | | |")
            continue
        g = lambda w, k: [r["w"][str(float(w))][k] for r in ev]
        L.append(f"| {cfg} | {len(ev)} | {tr[0]['n_params']:,} | {ms([t['train_minutes'] for t in tr], '.1f')} | "
                 f"{ms(g(0, 'fidelity'), '.3f')} | {ms(g(1, 'fidelity'), '.3f')} | {ms(g(1, 'class_match'), '.3f')} | "
                 f"{ms(g(1, 'variety_ratio'), '.3f')} | {ms(g(3, 'variety_ratio'), '.3f')} | {ms(g(1, 'nn_to_train'), '.2f')} |")
    L.append("")

    L += ["## Cost", ""]
    for sch in T0_SCHEDULES:
        L.append(f"- Tier 0 {sch} 학습: {ms([t['train_minutes'] for t in loads([f'tier0_{sch}_seed{s}_train.json' for s in SEEDS])], '.1f')} 분")
    L.append(f"- Tier 1 main 학습 ({T1_EPOCHS} epoch): {ms([t['train_minutes'] for t in loads([f'tier1_cosine_seed{s}_train.json' for s in SEEDS])], '.1f')} 분")
    # 보고서와 같은 출처: report_figures.py 의 step sweep (시드당 500장, CUDA 동기화). sample.py 로그(64장)는 쓰지 않음
    sweep = {}
    p = RESULTS / "figures" / "extra_numbers.json"
    if p.exists():
        sweep = json.load(open(p, encoding="utf-8")).get("A_t1_step_sweep_w1", {})
    if sweep and T1_SAMPLE_STEPS in sweep["steps"]:
        i = sweep["steps"].index(T1_SAMPLE_STEPS)
        secs = [row[i] for row in sweep["uniform"]["sec_per_100"]]
        L.append(f"- Tier 1 샘플링 ({T1_SAMPLE_STEPS}스텝, w=1, 시드당 500장): {ms(secs, '.2f')} 초 / 100샘플")
    else:
        L.append(f"- Tier 1 샘플링 ({T1_SAMPLE_STEPS}스텝, w=1): 미측정 (report_figures.py 실행 후 갱신)")
    L += ["", "샘플링 전용 실험(스텝 간격, guidance 구간 등)의 숫자: results/figures/extra_numbers.json"]

    (RESULTS / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


def report():
    collect()
    summarize()
    run("eda", RESULTS / "eda.json", "eda.py")      # Section 2 numbers and Figure 1
    print("[run ] report_figures", flush=True)
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / "report_figures.txt", "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "report_figures.py"], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        FAILED.append("report_figures")
        print("[FAIL] report_figures -> logs/report_figures.txt (그림 일부 실패)", flush=True)
    summarize()                                 # extra_numbers.json 이 새로 생겼을 수 있으니 요약을 다시 씀


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tier0()
    for s in SEEDS:
        tier1("main", s, T1_W, sample=True)
    report()                                    # main 만으로 보고서 재료부터 확보
    for cfg in ABLATIONS:
        for s in SEEDS:
            tier1(cfg, s, ABL_W)
        collect(); summarize()                  # ablation 하나 끝날 때마다 표 갱신
    report()
    print(f"\n완료 -> {RESULTS}/" + (f"   실패: {FAILED}" if FAILED else ""))
