# 디퓨전 토이 프로젝트 — 스타터 코드

CAS2250 인공지능개론 학기 프로젝트용입니다. 여러분이 채울 곳은 **두 군데**입니다.

| 파일 | 뭐가 들어 있나 |
|---|---|
| `diffusion.py` | 노이즈 스케줄, 순방향 과정, DDIM 한 스텝, 안전장치 — **그대로 쓰세요** |
| `data.py` | 2차원 나선(코드로 생성), MNIST 로더 — 그대로 |
| `models.py` | TimeMLP(2D), SmallUNet(28×28) — 그대로, 또는 diffusers로 교체 |
| `metrics.py` | 채점 대상 네 축 전부 — 그대로 (팀마다 다르게 재면 비교가 안 됩니다) |
| **`train.py`** | **TODO 1 — `training_step`** |
| **`sample.py`** | **TODO 2 — `sample_loop`** |
| `eval.py` | 네 축을 재서 `results.json` 으로 — 그대로 |

## 시작하기

```bash
pip install -r requirements.txt
python diffusion.py            # 안전장치 자체 점검. 여기부터 통과시키세요.
python data.py                 # 데이터가 [-1, 1] 인지 확인
python models.py               # 파라미터 수 확인 (SmallUNet 약 350만)
```

여기까지 다 돌면 환경은 준비된 겁니다.

## Tier 0 — 2차원 나선

```bash
python train.py  --tier 0 --steps 60000 --schedule cosine
python sample.py --ckpt runs/tier0_cosine_seed0.pt --steps 200 --n 3000
python eval.py   --ckpt runs/tier0_cosine_seed0.pt --steps 3 5 10 20 50 200
```

CPU로도 돕니다. 다만 1분으로는 안 됩니다 — 바닥에 도달하려면 8분쯤 걸립니다.
`eval.py` 가 찍어 주는 `d` 곡선이 보고서 필수 그림 중 하나입니다.

## Tier 1 — 28×28 MNIST

```bash
python train.py  --tier 1 --epochs 20 --schedule cosine
python sample.py --ckpt runs/tier1_cosine_seed0.pt --steps 50 --w 1.0
python eval.py   --ckpt runs/tier1_cosine_seed0.pt --steps 50 --w 0 1 2 3 5 8
```

GPU 권장입니다. Colab 무료 T4에서 20 epoch에 20~40분 정도 걸립니다.

## 막혔을 때

| 증상 | 십중팔구 원인 |
|---|---|
| 샘플이 전부 노이즈 | 데이터 정규화 누락, 또는 ᾱ가 0까지 안 내려감. `python diffusion.py` |
| 결과가 회색 얼룩 | `model(x, t)` 에 `t`를 안 넘겼습니다 |
| 샘플에 NaN | ᾱ 클램프 또는 x0 클리핑 누락 — `ddim_step` 을 쓰면 자동으로 잡힙니다 |
| 스텝을 늘릴수록 나빠짐 | `ddim_step` 을 직접 다시 짜면서 eps 재유도를 빠뜨린 경우 |
| 손실이 안 내려감 | 정상입니다. 디퓨전 손실은 일찍 평평해집니다. 샘플 그리드를 보세요 |
| 분류기 정확도 assert | epoch를 늘리세요. 98% 미만이면 fidelity 숫자를 인정하지 않습니다 |

## 규칙

- 학습된 **디퓨전 체크포인트를 받아다 제출하는 것은 안 됩니다.** 구조만 빌려 오는 건 괜찮습니다.
- diffusers 의 스케줄러나 파이프라인을 그대로 호출해서 샘플링하는 것도 안 됩니다.
  `training_step` 과 `sample_loop` 이 채점 대상입니다.
- `ddim_step` 은 저희가 드립니다. 수치적으로 위험한 부분이라 모두가 같은 조건에서
  측정하도록 고정했습니다. 여러분이 쓰실 건 그걸 엮는 루프와 guidance 혼합입니다.
- 시드는 최소 세 개 돌리고, 평균과 표준편차를 같이 보고하세요.

## 제출물

제출 파일 하나로 압축해서 내세요. 압축 파일 이름은 `<학번>_task2_diffusion.zip` 입니다.

```
<학번>_task2_diffusion/
├── report.pdf                        보고서 1~2쪽
├── code/                             스타터 폴더 전체 (TODO 를 채운 것)
│   ├── train.py                      <- 여러분이 채운 파일
│   ├── sample.py                     <- 여러분이 채운 파일
│   └── (나머지 7개는 받은 그대로)
└── results/
    ├── results_tier0_seed*.json      eval.py 출력 (시드 3개)
    ├── results_tier1_seed*.json      Tier 1 까지 갔다면
    ├── grid_8x8.png                  큐레이션하지 않은 고정 시드 그리드
    ├── grids/ep*.png                 epoch 별 시계열 (train.py 가 자동 저장)
    └── *_train.json                  train.py 가 남긴 학습 기록
```

linear 와 cosine 어블레이션은 `results_tier0_*.json` 이 두 스케줄 모두 있으면 됩니다.
`--schedule linear` 로 한 번 더 돌리세요.

### 넣지 마세요

| 빼야 할 것 | 이유 |
|---|---|
| `runs/*.pt` 체크포인트 | 채점에 쓰지 않습니다 |
| `data/` | MNIST 입니다. 코드가 알아서 받습니다 |
| `__pycache__/`, `.venv/` | |

체크포인트는 채점에 쓰지 않습니다. `results.json` 과 코드로 재현합니다.
그래서 **README 에 적은 명령이 그대로 돌아가는 것** 이 중요합니다.

## 제출 전 체크

- [ ] 위 '제출물' 구조대로 묶었고, 체크포인트(`.pt`)와 `data/` 를 뺐는가
- [ ] `pip install -r requirements.txt` 만으로 환경이 재현되는가
- [ ] README에 적은 명령을 그대로 복사해서 붙이면 도는가
- [ ] 보고서의 모든 숫자가 그 명령들의 출력에서 나왔는가
- [ ] 큐레이션하지 않은 8×8 그리드와, 같은 시드의 epoch별 시계열이 들어 있는가
- [ ] 스텝-품질 곡선을 추측이 아니라 측정했는가 (시드 3개)
- [ ] linear vs cosine 어블레이션이 들어 있는가
- [ ] 네 축(fidelity / coverage + variety / memorization / cost) 숫자가 다 있는가
