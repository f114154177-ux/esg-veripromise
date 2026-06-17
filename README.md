# ESG VeriPromise — Calibrated RoBERTa Submission (Public LB 0.5697470)

NKUST 類神經網路期末報告 · 企業永續報告書 ESG 承諾驗證競賽 (VeriPromiseESG, AIdea)

**Team:** TEAM_10080 — F114154171 張維鵬 · F114154174 簡宏宇 · F114154177 焦聖崴

| Metric | Score |
|---|---|
| AIdea Public Leaderboard | **0.5697470** (Rank 101/131) |
| Baseline checkpoint (public) | 0.5646062 |
| Calibrated local validation (`val_200`) | 0.60211 |
| Public gain from calibration | +0.0051408 |

> Note: local validation (0.60211) overestimates the public score (0.5697470) by ~0.0324. This is expected validation overfit on a 200-sample split, but the calibrated model still improves on the 0.5646062 baseline.

---

## Task

Given a Traditional-Chinese ESG paragraph from a corporate sustainability report, predict four labels:

| Field | Classes |
|---|---|
| `promise_status` | `Yes`, `No` |
| `verification_timeline` | `already`, `within_2_years`, `between_2_and_5_years`, `more_than_5_years`, `N/A` |
| `evidence_status` | `Yes`, `No`, `N/A` |
| `evidence_quality` | `Clear`, `Not Clear`, `Misleading`, `N/A` |

The submission CSV has columns `id, promise_status, verification_timeline, evidence_status, evidence_quality`, written as **UTF-8 (no BOM)** with **Unix LF** line endings.

---

## Method

```
hfl/chinese-roberta-wwm-ext encoder
        ↓  (pooler_output / CLS)
Dropout(0.1)
        ↓
4 independent linear heads  →  promise_status / verification_timeline / evidence_status / evidence_quality
        ↓
+ fixed logit-bias calibration
        ↓
+ rule-based post-processing
        ↓
submission.csv
```

1. **Multi-task RoBERTa.** A single `hfl/chinese-roberta-wwm-ext` encoder is shared across four independent linear classification heads, so the tasks share ESG semantic features while keeping separate label spaces.

2. **Baseline checkpoint.** Trained with supervised fine-tuning on `train_800.json` / `val_200.json` (1000 labelled rows), 8 epochs, max length 256, batch size 8, lr 2e-5, weight decay 0.01, warmup ratio 0.1, dropout 0.1, per-task `CrossEntropyLoss` with inverse-frequency class weights. Best checkpoint selected by validation weighted macro-F1. This baseline scores **0.5646062** on the public LB.

3. **Logit-bias calibration (the actual improvement).** Instead of retraining, a fixed bias vector is added to each task's logits before `argmax`. This mainly fixes `verification_timeline`, which was over-predicting `more_than_5_years`. The exact bias (`calibration_candidates/calibrated_all_full_biases.json`):

   ```json
   {
     "promise_status":        [0.6, 0.0],
     "verification_timeline": [0.4, 0.8, 0.8, -0.2, 0.4],
     "evidence_status":       [0.8, 0.0, 0.0],
     "evidence_quality":      [0.15, 0.0, 0.0, 0.0]
   }
   ```
   Effect: boost `promise=Yes`; boost timeline `already / within_2_years / between_2_and_5_years / N/A` and lower `more_than_5_years`; boost `evidence_status=Yes`; slightly boost `evidence_quality=Clear`.

4. **Rule-based post-processing** enforces the label hierarchy:
   - `promise_status = No` → all three downstream fields forced to `N/A`.
   - `promise_status = Yes` with implausible `N/A`: timeline `N/A` → `between_2_and_5_years`; `evidence_status N/A` → `No`; `evidence_status No` → `evidence_quality N/A`; `evidence_status Yes` with `evidence_quality N/A` → `Clear`.

Internally the checkpoint uses the label `longer_than_5_years`; it is mapped to the official `more_than_5_years` only when writing the CSV.

**Not used** (tested but did not stably help): MacBERT (local val only 0.53964), data augmentation, ensembling, pseudo-labels, LLM few-shot, and the `promise_string` / `evidence_string` fields (excluded to avoid label leakage).

---

## Repository layout

```
.
├── calibrate_baseline.py                 # load checkpoint → calibrate → write submission.csv
├── requirements.txt
├── submission.csv                        # the 0.5697470 submission (2000 rows)
├── calibration_candidates/
│   └── calibrated_all_full_biases.json   # the fixed logit bias
└── data/
    ├── val_200.json                      # 200 labelled validation rows
    ├── vpesg4k_test_2000.json            # 2000 unlabelled test rows
    └── sample_submission_format.csv      # official format reference
```

The trained checkpoint (`best_model.pt`) is **not** included — it is a large binary file. See below.

---

## Setup

Requires **Python 3.12**. A CUDA GPU is recommended but the script falls back to CPU.

```bash
pip install -r requirements.txt
```

### Provide the checkpoint

The script needs the baseline RoBERTa checkpoint (`best_model.pt`, plus an optional `tokenizer/` folder next to it). Either:

- place it at `./checkpoints/best_model.pt` (default), **or**
- point the `ESG_CHECKPOINT` environment variable at its path:

  ```bash
  export ESG_CHECKPOINT=/path/to/best_model.pt
  ```

The checkpoint is a `torch.save` dict containing `args` (with `model_name`, `max_len`, `dropout`) and `model_state_dict`. If a `tokenizer/` directory sits beside the checkpoint it is used; otherwise the tokenizer is downloaded from `model_name` via Hugging Face.

---

## 操作流程(逐步說明)

**步驟 1 — 安裝套件**

```bash
pip install -r requirements.txt
```

**步驟 2 — 放好 checkpoint**(詳見上方 [Setup](#setup))

```bash
# 方法 A:把 best_model.pt 複製到 ./checkpoints/best_model.pt
# 方法 B:用環境變數指定路徑
export ESG_CHECKPOINT=/path/to/best_model.pt
```

**步驟 3 — 執行主程式**

```bash
python calibrate_baseline.py
```

**步驟 4 — 取得輸出。** `submission.csv` 會產生在腳本同一層資料夾,可直接上傳到 AIdea。終端機最後會印出:

```
Saved: .../submission.csv
Local validation weighted score: 0.60211
  promise_status           ...
  verification_timeline    ...
  evidence_status          ...
  evidence_quality         ...
```

本 repo 內附的 `submission.csv` 就是在 AIdea 公開排行榜得到 **0.5697470** 的那一份。

### 程式內部執行流程

執行 `calibrate_baseline.py` 時,`main()` 會依序跑完以下流程:

| # | 階段 | 對應程式 |
|---|---|---|
| 1 | 設定隨機種子(42),確保可重現 | `set_seed()` |
| 2 | 載入 checkpoint,從其 `args` 讀取 `model_name`、`max_len`、`dropout` | `torch.load(...)` |
| 3 | 建立多任務模型並載入權重 | `MultiTaskESGModel`、`load_state_dict` |
| 4 | 載入固定的 logit bias | `load_biases()` |
| 5 | 讀取 `val_200.json` / `vpesg4k_test_2000.json`,把 `more_than_5_years` 轉成內部用的 `longer_than_5_years` | `load_json()` |
| 6 | 前向推論 → 各任務 logits | `extract_logits()` |
| 7 | 加上 bias、取 `argmax`、套用階層規則 | `predict_from_logits()` → `apply_rules()` |
| 8 | 對 `val_200` 算分(weighted macro-F1,約 0.60211)並印出各欄位 F1 | `evaluate()` |
| 9 | 把標籤轉回官方名稱並寫出 CSV | `submission_timeline()`、`write_submission()` |
| 10 | 驗證:2000 列、欄位順序正確、UTF-8 無 BOM | `validate_submission()` |

> 注意:這支程式只做**推論 + 校正**,它載入已訓練好的 checkpoint,**不會重新訓練模型**。模型架構定義在 `MultiTaskESGModel` 這個 class;產生 `best_model.pt` 的 supervised fine-tuning 訓練程式並不包含在這支腳本內。

---

## References

- [1] VeriPromiseESG / AIdea — ESG Promise Verification Competition dataset and submission format. <https://aidea-web.tw/>
- [2] Hugging Face — `hfl/chinese-roberta-wwm-ext`. <https://huggingface.co/hfl/chinese-roberta-wwm-ext>
- [3] Y. Cui et al., "Pre-Training with Whole Word Masking for Chinese BERT," 2019.
- [4] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," EMNLP System Demonstrations, 2020.
- [5] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS, 2019.
- [6] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL-HLT, 2019.

## Generative AI disclosure

Generative AI tools were used to help inspect files, debug the inference/calibration script, verify CSV format, and draft documentation. AI tools were **not** used to obtain hidden test labels and did not use any non-public label information. All model training, inference, submission checks, and AIdea uploads were verified by team members.
