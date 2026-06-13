# IFD 기반 데이터 선별을 통한 데이터 효율적 LLM Fine-tuning

Cherry LLM(NAACL'24)의 **IFD(Instruction-Following Difficulty) 기반 데이터 선별** 기법을 직접 읽고 **Qwen2.5-1.5B**에 구현·검증한 프로젝트입니다. 핵심 질문은 다음과 같습니다.

> *잘 고른 소수의 instruction 데이터만으로 전체 데이터 학습에 필적하거나 능가할 수 있는가?*

이 프로젝트는 석사 논문(**SemiPEP**)에서 착안했습니다. SemiPEP는 semi-supervised 환경에서 confidence를 기준으로 신뢰할 수 있는 데이터를 선별해 학습했는데, "전부 쓰지 말고 좋은 데이터를 고른다"는 철학이 최근 LLM 데이터 선별 연구와 맞닿아 있습니다. 본 프로젝트는 두 가지 선별 신호 — **IFD(어렵고 유익한 샘플)** 와 **confidence(쉽고 신뢰할 수 있는 샘플)** — 를 비교합니다.

---

## 결과 요약

ARC-Challenge (0-shot), Qwen2.5-1.5B + LoRA, 선별 비율 20%:

| 모델 | 선별 방식 | ARC-C | HellaSwag |
|------|-----------|:-----:|:---------:|
| Baseline | 학습 안 함 | 45.22 | 67.79 |
| A | 전체 100% (51K) | 47.53 | 67.67 |
| B | 랜덤 20% | 47.95 | 67.50 |
| **C** | **IFD 상위 20%** | **48.29** | 67.18 |
| D | confidence 상위 20% | 45.99 | 67.46 |

**핵심 발견 (ARC-Challenge 기준):**
- **IFD로 고른 20%(48.29)가 전체 100% 학습(47.53)을 능가** — 데이터 효율성 확인
- **IFD(48.29) > 랜덤(47.95)** — 단순히 데이터가 적어서가 아니라 선별 신호 자체가 유효
- **confidence 기반 "쉬운" 선별(45.99) ≈ baseline(45.22)** — 사전학습된 모델에 쉬운 샘플 선별은 거의 효과 없음

HellaSwag는 조건 간 유의미한 차이가 없었고, ARC-Challenge가 변별력 있는 벤치마크였습니다. 1.5B 소형 모델 + 0-shot 환경 특성상 차이는 작지만(~1점), 순위는 일관되고 해석 가능합니다.

---

## 동기

석사 논문 **SemiPEP**는 confidence로 신뢰할 수 있는 pseudo-label을 선별해 semi-supervised 학습을 안정화했습니다(신뢰도 높은 것부터 시작해 점진적으로 확장하는 커리큘럼 방식). "전부 쓰지 말고 좋은 데이터를 고른다"는 핵심 아이디어는 최근 LLM 연구와 직접 연결됩니다.

- **Cherry LLM** (NAACL'24) — IFD score로 instruction 데이터 선별, 약 10%로 100%에 필적
- **Superfiltering** (ACL'24) — 작은 모델로도 IFD를 효과적으로 계산 가능
- **LIMA, AlpaGasus** — 데이터 양보다 품질이 중요

이 아이디어를 RTX 4070 Ti 단일 GPU에서 돌릴 수 있는 LLM instruction-tuning 환경으로 옮겼습니다.

---

## 방법론

### IFD score (Cherry LLM)
```
IFD = mean_CE(response | instruction) / mean_CE(response 단독)
```
IFD가 높을수록 instruction이 생성에 별로 도움이 안 됨 = 어렵고 유익한 샘플. base Qwen2.5-1.5B로 계산(학습 없이 forward pass만). IFD > 1인 샘플은 이상치로 간주해 제외.

### Confidence score (SemiPEP에서 차용)
```
Global    = exp(-mean_CE(response | instruction))
Local     = exp(-mean_CE(가장 어려운 25% 토큰의 CE))
Combined  = sqrt(Global x Local)
```
confidence가 높을수록 모델이 이미 쉽게 생성 = 쉽고 신뢰할 수 있는 샘플. IFD와 confidence는 음의 상관(Alpaca에서 Spearman rho ≈ -0.44)을 보입니다. IFD 높은 샘플은 어렵고(confidence 낮음), 그 반대도 성립.

### 비교군 (Qwen2.5-1.5B + LoRA, 동일 하이퍼파라미터)
| 그룹 | 선별 방식 | 의도 |
|------|-----------|------|
| A | 전체 100% | 상한 baseline |
| B | 랜덤 20% | 하한 baseline |
| C | IFD 상위 20% | Cherry LLM (어려운 샘플) |
| D | confidence 상위 20% | SemiPEP 방식 (쉬운 샘플) |

평가: EleutherAI **lm-evaluation-harness** (ARC-Challenge, HellaSwag). Cherry LLM이 사용한 벤치마크 계열을 로컬에서 API 없이 실행.

---

## 여기까지의 과정: 왜 GSM8K가 아니라 Alpaca인가

이 파이프라인은 처음에 **GSM8K + 노이즈 주입**으로 시작했다가 폐기했습니다. 그 이유를 투명하게 남겨두기 위해 관련 결과를 `outputs/gsm8k_*`에 보존했습니다.

**1. Qwen2.5가 주입된 노이즈를 흡수했다.** 50% 노이즈 vs clean GSM8K 학습이 *동일한* 최종 정확도(57.1% vs 57.1%)를 보임. 모델의 강력한 사전학습 prior가 노이즈를 덮어씀.

**2. GSM8K 점수가 너무 균일해서 선별이 불가능했다.** IFD std = 0.10(0.4~0.6에 밀집), IFD-confidence Spearman rho = -0.62(신호가 거의 중복). 선별이 의미를 가질 여지가 없음.

**3. Alpaca는 선별에 적합했다.** IFD std = 0.23(2배 넓음), rho(IFD, confidence) = -0.44(더 독립적), IFD > 1 이상치 약 675개 존재. Alpaca는 Cherry LLM 원논문의 데이터셋이기도 해서 직접 비교가 가능.

GSM8K 실패를 정량적으로 진단하고 방향을 전환한 것이 이 프로젝트 학습의 핵심이었습니다.

---

## confidence 기반 선별(D)이 실패한 이유

SemiPEP의 커리큘럼(쉬운 것부터)이 효과적이었던 건 **from scratch** 학습이기 때문입니다. 아무것도 모르는 모델에 처음부터 어렵거나 노이즈가 섞인 라벨을 주면 학습이 불안정해집니다. 하지만 이번엔 모델이 **이미 사전학습**되어 있습니다. "쉬운" 샘플은 모델이 이미 아는 것이라 학습에 보탬이 적습니다. 결과가 이를 확인합니다 — D(쉬운 선별)는 baseline에서 거의 움직이지 않은 반면, C(어렵고 유익한 선별)가 가장 효과적이었습니다.

**교훈: scratch 학습에서 통하던 easy-first 커리큘럼 선별이, 강력한 사전학습 모델의 fine-tuning에는 그대로 전이되지 않는다.**

---

## 실행 환경

- **GPU:** RTX 4070 Ti (12 GB)
- **모델:** Qwen2.5-1.5B (base) + LoRA (r=16, alpha=32, dropout=0.05)
- **스택:** Python 3.10, PyTorch 2.6 (CUDA 12.4), Transformers, PEFT, lm-eval
- **데이터:** Alpaca 52K

---

## 저장소 구조

```
.
├── src/
│   ├── data/            # Alpaca / GSM8K 로더, 프롬프트 포맷, 노이즈(legacy)
│   └── evaluation/      # 정확도 / 메트릭 helper
├── scripts/
│   ├── compute_alpaca_scores.py   # IFD + confidence 계산 (Alpaca 52K)
│   ├── prepare_alpaca_splits.py   # A/B/C/D 선별 세트 구성
│   ├── train_alpaca.py            # Qwen2.5-1.5B + LoRA 학습
│   ├── summarize_lmeval.py        # lm-eval 결과 표로 정리
│   └── ...                        # GSM8K 스크립트 (초기 시도)
├── configs/             # 실험별 YAML 설정
├── outputs/
│   ├── alpaca_*         # 메인 실험 (최종 결과)
│   ├── alpaca_scores/   # IFD / confidence 점수
│   ├── eval_results/    # lm-eval 벤치마크 결과
│   └── gsm8k_*, smoke_*, stage1_*   # 초기 시도, 투명성을 위해 보존
└── README.md
```

참고: `reference/SemiPEP`(석사 논문 코드)와 `reference/Cherry_LLM`은 본 저장소에 포함하지 않았습니다. 출처는 하단 References 참고.

---

## 재현 방법

```bash
# 1. 전체 Alpaca 샘플 점수 계산 (IFD + confidence)
python scripts/compute_alpaca_scores.py

# 2. A/B/C/D 선별 세트 구성
python scripts/prepare_alpaca_splits.py

# 3. 학습 (그룹별로 config 바꿔가며 반복)
python scripts/train_alpaca.py --config configs/alpaca_config.yaml

# 4. lm-eval로 평가 (모델별)
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen2.5-1.5B,peft=outputs/alpaca_C_ifd/final_model,dtype=bfloat16 \
  --tasks arc_challenge,hellaswag --device cuda:0 --batch_size 8

# 5. 결과 정리
python scripts/summarize_lmeval.py
```

---

## 한계 및 정직한 범위

이 프로젝트는 **재현·검증 연구**이지 새로운 방법론 제안이 아닙니다. 1.5B 모델 + 0-shot + LoRA 환경이라 효과 크기가 작으므로, 결과는 통계적으로 강한 차이라기보다 *방향성*을 보여줍니다. 평가는 0-shot 기준(Cherry LLM의 25/10-shot이 아님)이라 원논문과 절대 수치를 직접 비교할 수 없으며, 본 실험 내 조건들 간의 상대 비교만 유효합니다.

## 향후 과제

- confidence를 "쉬운 것 선택"이 아니라 **저품질/노이즈 필터링**(나쁜 샘플 제거)으로 재정의
- 신뢰할 수 있는 샘플에서 어려운 샘플로 점진적으로 확장하는 **진짜 커리큘럼** 구현으로 순서 효과 분리
- 더 큰 모델(Qwen2.5-3B/7B)에서 선별 효과가 규모에 따라 커지는지 검증

---

## References

- Li et al. *From Quantity to Quality: Self-Guided Data Selection for Instruction Tuning* (Cherry LLM), NAACL 2024.
- Li et al. *Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning*, ACL 2024.
- Zhou et al. *LIMA: Less Is More for Alignment*, 2023.
- SemiPEP — 석사 논문 (confidence 기반 semi-supervised 선별).
