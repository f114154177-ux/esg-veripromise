# ESG VeriPromise — 校正式多任務 RoBERTa(公開排行榜 0.5697470)

NKUST 類神經網路期末報告 · 企業永續報告書 ESG 承諾驗證競賽(VeriPromiseESG, AIdea)

**隊伍:** TEAM_10080 — F114154171 張維鵬 · F114154174 簡宏宇 · F114154177 焦聖崴

| 指標 | 分數 |
|---|---|
| AIdea 公開排行榜 | **0.5697470**(Rank 101/131) |
| Baseline checkpoint(公開分數) | 0.5646062 |
| 校正後本地驗證(`val_200`) | 0.60211 |
| 校正帶來的公開分數提升 | +0.0051408 |

> 說明:本地驗證(0.60211)比公開分數(0.5697470)高約 0.0324,屬於在 200 筆驗證集上調參的 overfit 現象;但校正版仍比 0.5646062 baseline 高。

---

## 一、任務說明

輸入一段企業永續報告書的繁體中文 ESG 段落,預測四個標籤:

| 欄位 | 類別 |
|---|---|
| `promise_status` | `Yes`, `No` |
| `verification_timeline` | `already`, `within_2_years`, `between_2_and_5_years`, `more_than_5_years`, `N/A` |
| `evidence_status` | `Yes`, `No`, `N/A` |
| `evidence_quality` | `Clear`, `Not Clear`, `Misleading`, `N/A` |

提交檔(submission CSV)欄位為 `id, promise_status, verification_timeline, evidence_status, evidence_quality`,以 **UTF-8(無 BOM)**、**Unix LF** 換行寫出。

---

## 二、方法概述

```
hfl/chinese-roberta-wwm-ext encoder
        ↓  (pooler_output / CLS)
Dropout(0.1)
        ↓
四個獨立 linear heads → promise_status / verification_timeline / evidence_status / evidence_quality
        ↓
+ 固定 logit-bias 校正
        ↓
+ rule-based 後處理
        ↓
submission.csv
```

1. **多任務 RoBERTa**:共用一個 `hfl/chinese-roberta-wwm-ext` encoder,接四個獨立 linear 分類頭,四個任務共享 ESG 語意特徵但各自保留獨立標籤空間。
2. **Baseline checkpoint**:以 `train_800.json` / `val_200.json`(共 1000 筆)做 supervised fine-tuning,每個任務用 `CrossEntropyLoss` + inverse-frequency class weight,依驗證 weighted macro-F1 保存最佳 checkpoint。此 baseline 公開分數為 **0.5646062**。
3. **Logit-bias 校正(真正的提升來源)**:不重新訓練,而是在每個任務取 `argmax` 前,對 logits 加上一組固定 bias,主要修正過度偏向 `more_than_5_years` 的 `verification_timeline`。bias 來自 `calibration_candidates/calibrated_all_full_biases.json`:

   ```json
   {
     "promise_status":        [0.6, 0.0],
     "verification_timeline": [0.4, 0.8, 0.8, -0.2, 0.4],
     "evidence_status":       [0.8, 0.0, 0.0],
     "evidence_quality":      [0.15, 0.0, 0.0, 0.0]
   }
   ```
   效果:提高 `promise=Yes`;提高 timeline 的 `already / within_2_years / between_2_and_5_years / N/A` 並降低 `more_than_5_years`;提高 `evidence_status=Yes`;略為提高 `evidence_quality=Clear`。
4. **Rule-based 後處理**(維持標籤階層邏輯):
   - `promise_status = No` → 下游三欄全部強制 `N/A`。
   - `promise_status = Yes` 但下游出現不合理 `N/A`:timeline `N/A` → `between_2_and_5_years`;`evidence_status N/A` → `No`;`evidence_status No` → `evidence_quality N/A`;`evidence_status Yes` 且 `evidence_quality N/A` → `Clear`。

模型內部使用 `longer_than_5_years` 這個標籤,只有在寫出 CSV 時才轉成官方的 `more_than_5_years`。

**沒有採用**(測試過但無法穩定提升):MacBERT(本地驗證僅 0.53964)、資料增強、ensemble、pseudo-label、LLM few-shot,以及 `promise_string` / `evidence_string`(避免答案洩漏)。

---

## 三、檔案結構

```
.
├── train_and_submit_baseline.py          # 【訓練】訓練 baseline → 產生 checkpoint 與 baseline submission
├── calibrate_baseline.py                 # 【推論+校正】載入 checkpoint + 固定 bias → 產生最終 submission.csv
├── requirements.txt
├── submission.csv                        # 公開分數 0.5697470 的提交檔(2000 列)
├── calibration_candidates/
│   └── calibrated_all_full_biases.json   # 固定 logit bias
└── data/
    ├── vpesg4k_train_1000.json           # 原始 1000 筆標註資料
    ├── train_800.json                    # 訓練集(800 筆)
    ├── val_200.json                      # 驗證集(200 筆)
    ├── vpesg4k_test_2000.json            # 測試集(2000 筆,無標註)
    └── sample_submission_format.csv      # 官方格式範例
```

> **關於模型權重檔**:訓練好的 checkpoint `best_model.pt` 約 **391MB,超過 GitHub 單檔 100MB 上限,因此不放在 repo 內**(`.gitignore` 已排除)。你可以用下方「路徑 A」自行訓練產生它,或用 Git LFS / 雲端連結另外提供(見第六節)。

---

## 四、環境安裝

需要 **Python 3.12**,建議有 CUDA GPU(沒有也能在 CPU 上跑,但訓練很慢)。

```bash
pip install -r requirements.txt
```

---

## 五、操作流程(逐步說明)

本專案有兩條路徑。**路徑 A** 從頭訓練出 checkpoint;**路徑 B** 用 checkpoint 加固定 bias 產生最終提交檔。完整重現 = A → B。

### 路徑 A:訓練 baseline(產生 checkpoint)

```bash
python train_and_submit_baseline.py --mode train_submit
```

預設會自動讀取 `data/` 內的資料,並輸出到 `outputs/`:

```
outputs/checkpoints/best_model.pt        # 訓練好的權重(校正會用到)
outputs/checkpoints/tokenizer/           # tokenizer
outputs/submission_baseline.csv          # 未校正的 baseline 提交檔
outputs/training_history.json            # 每個 epoch 的分數紀錄
outputs/prediction.json                  # 驗證集預測
```

可用參數(預設值即為本專案設定):

```bash
python train_and_submit_baseline.py --mode train_submit \
  --train-path data/train_800.json \
  --val-path data/val_200.json \
  --test-path data/vpesg4k_test_2000.json \
  --epochs 8 --batch-size 8 --max-len 256 --lr 2e-5 \
  --weight-decay 0.01 --warmup-ratio 0.1 --dropout 0.1
```

- `--mode train`:只訓練、不產生提交檔。
- `--mode submit --checkpoint <path>`:只用既有 checkpoint 產生提交檔。
- `--mode train_submit`:訓練後直接產生 baseline 提交檔(預設)。

> 注意:重新訓練因 GPU、套件版本與隨機性,結果可能和原 checkpoint 略有差異。官方公開分數 0.5697470 是以原始 checkpoint + 固定 bias 推論得到。

### 路徑 B:校正推論(產生最終 0.5697470 提交檔)

校正程式預設到 `./checkpoints/best_model.pt` 找權重,或用環境變數 `ESG_CHECKPOINT` 指定。把路徑 A 訓練出的 checkpoint 接上即可:

```bash
export ESG_CHECKPOINT=outputs/checkpoints/best_model.pt
python calibrate_baseline.py
```

輸出 `submission.csv`(寫在腳本同層),終端機最後會印出:

```
Saved: .../submission.csv
Local validation weighted score: 0.60211
  promise_status           ...
  verification_timeline    ...
  evidence_status          ...
  evidence_quality         ...
```

repo 內附的 `submission.csv` 就是 AIdea 公開排行榜得到 **0.5697470** 的那一份。

### `calibrate_baseline.py` 內部流程

| # | 階段 | 對應程式 |
|---|---|---|
| 1 | 設定隨機種子(42) | `set_seed()` |
| 2 | 載入 checkpoint,讀 `model_name`、`max_len`、`dropout` | `torch.load(...)` |
| 3 | 建立多任務模型並載入權重 | `MultiTaskESGModel`、`load_state_dict` |
| 4 | 載入固定 logit bias | `load_biases()` |
| 5 | 讀資料,`more_than_5_years` → 內部 `longer_than_5_years` | `load_json()` |
| 6 | 前向推論 → 各任務 logits | `extract_logits()` |
| 7 | 加 bias、取 `argmax`、套階層規則 | `predict_from_logits()` → `apply_rules()` |
| 8 | 對 `val_200` 算分(約 0.60211) | `evaluate()` |
| 9 | 標籤轉回官方名稱、寫 CSV | `write_submission()` |
| 10 | 驗證:2000 列、欄位正確、UTF-8 無 BOM | `validate_submission()` |

---

## 六、(選用)把 checkpoint 一起發布

若希望別人不必重新訓練就能跑路徑 B,有兩種做法:

- **GitHub Release / 雲端連結**:把 `best_model.pt` 與 `tokenizer/` 打包上傳到 GitHub Release 或 Google Drive / Hugging Face Hub,在此處附下載連結。
- **Git LFS**:`git lfs install && git lfs track "*.pt"`,再把 checkpoint 加入版本控制(需 LFS 額度)。

下載後放到 `./checkpoints/best_model.pt`(tokenizer 放在 `./checkpoints/tokenizer/`)即可直接執行路徑 B。

---

## 七、主要訓練參數

| 參數 | 值 |
|---|---|
| encoder | `hfl/chinese-roberta-wwm-ext` |
| epochs | 8 |
| max length | 256 |
| batch size | 8 |
| learning rate | 2e-5 |
| weight decay | 0.01 |
| warmup ratio | 0.1 |
| dropout | 0.1 |
| loss | 四任務各自 `CrossEntropyLoss` + inverse-frequency class weight |
| checkpoint 選擇 | 依驗證 weighted macro-F1 |

---

## 八、參考文獻

- [1] VeriPromiseESG / AIdea — ESG 承諾驗證競賽資料集與提交格式。<https://aidea-web.tw/>
- [2] Hugging Face — `hfl/chinese-roberta-wwm-ext`。<https://huggingface.co/hfl/chinese-roberta-wwm-ext>
- [3] Y. Cui et al., "Pre-Training with Whole Word Masking for Chinese BERT," 2019.
- [4] T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," EMNLP System Demonstrations, 2020.
- [5] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS, 2019.
- [6] J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," NAACL-HLT, 2019.

## 九、生成式 AI 使用揭露

本專案使用生成式 AI 工具協助檢視檔案、除錯推論與校正腳本、檢查 CSV 格式,以及整理文件草稿。AI 工具**未**用於取得測試集答案,也未使用任何未公開標籤資訊。所有模型訓練、推論、submission 檢查與 AIdea 上傳結果皆由團隊成員確認。
