#!/usr/bin/env python3
"""Reproduce the 0.60211 local-validation calibrated ESG submission.

This is the cleaned final path:
- load the known-good RoBERTa checkpoint
- apply the fixed logit bias that scored 0.60211 on val_200
- write submission.csv next to this script

It intentionally does not run a parameter sweep, so rerunning it will not
recreate the old clutter of many candidate CSV files.

Paths are resolved relative to this file so the project is portable. The
checkpoint location can be overridden with the ESG_CHECKPOINT environment
variable; otherwise it defaults to ./checkpoints/best_model.pt.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CANDIDATE_DIR = ROOT / "calibration_candidates"

# The baseline checkpoint is a large file and is NOT included in this repo.
# Place it at ./checkpoints/best_model.pt or set ESG_CHECKPOINT to its path.
CHECKPOINT = Path(os.environ.get("ESG_CHECKPOINT", ROOT / "checkpoints" / "best_model.pt"))
TOKENIZER_DIR = CHECKPOINT.parent / "tokenizer"
BIAS_PATH = CANDIDATE_DIR / "calibrated_all_full_biases.json"
VAL_PATH = DATA_DIR / "val_200.json"
TEST_PATH = DATA_DIR / "vpesg4k_test_2000.json"
SUBMISSION_PATH = ROOT / "submission.csv"

FIELDS = ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]
LABELS = {
    "promise_status": ["Yes", "No"],
    "verification_timeline": [
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "longer_than_5_years",
        "N/A",
    ],
    "evidence_status": ["Yes", "No", "N/A"],
    "evidence_quality": ["Clear", "Not Clear", "Misleading", "N/A"],
}
WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}
ID2LABEL = {field: {idx: label for idx, label in enumerate(labels)} for field, labels in LABELS.items()}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        if row.get("verification_timeline") == "more_than_5_years":
            row["verification_timeline"] = "longer_than_5_years"
    return rows


class ESGDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        encoding = self.tokenizer(
            row["data"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def collate(batch: List[dict]) -> dict:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }


class MultiTaskESGModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleDict({
            field: nn.Linear(hidden_size, len(LABELS[field]))
            for field in FIELDS
        })

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = (
            outputs.pooler_output
            if getattr(outputs, "pooler_output", None) is not None
            else outputs.last_hidden_state[:, 0]
        )
        pooled = self.dropout(pooled)
        return {field: head(pooled) for field, head in self.classifiers.items()}


def apply_rules(pred: Dict[str, str]) -> Dict[str, str]:
    pred = dict(pred)
    if pred["promise_status"] == "No":
        pred["verification_timeline"] = "N/A"
        pred["evidence_status"] = "N/A"
        pred["evidence_quality"] = "N/A"
        return pred
    if pred["verification_timeline"] == "N/A":
        pred["verification_timeline"] = "between_2_and_5_years"
    if pred["evidence_status"] == "N/A":
        pred["evidence_status"] = "No"
    if pred["evidence_status"] == "No":
        pred["evidence_quality"] = "N/A"
    elif pred["evidence_status"] == "Yes" and pred["evidence_quality"] == "N/A":
        pred["evidence_quality"] = "Clear"
    return pred


@torch.no_grad()
def extract_logits(model, tokenizer, rows: List[dict], device: torch.device, max_len: int) -> Dict[str, torch.Tensor]:
    loader = DataLoader(
        ESGDataset(rows, tokenizer, max_len),
        batch_size=32,
        shuffle=False,
        collate_fn=collate,
    )
    model.eval()
    chunks = {field: [] for field in FIELDS}
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        for field in FIELDS:
            chunks[field].append(logits[field].detach().cpu())
    return {field: torch.cat(parts, dim=0) for field, parts in chunks.items()}


def load_biases(path: Path) -> Dict[str, torch.Tensor]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {field: torch.tensor(raw[field], dtype=torch.float) for field in FIELDS}


def predict_from_logits(logits: Dict[str, torch.Tensor], biases: Dict[str, torch.Tensor]) -> List[Dict[str, str]]:
    rows = []
    size = next(iter(logits.values())).shape[0]
    for idx in range(size):
        pred = {}
        for field in FIELDS:
            scores = logits[field][idx] + biases[field]
            pred[field] = ID2LABEL[field][int(scores.argmax())]
        rows.append(apply_rules(pred))
    return rows


def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def evaluate(gold_rows: List[dict], pred_rows: List[Dict[str, str]]) -> Tuple[float, Dict[str, float]]:
    field_scores = {}
    weighted = 0.0
    for field in FIELDS:
        y_true = [row[field] for row in gold_rows]
        y_pred = [row[field] for row in pred_rows]
        score = macro_f1(y_true, y_pred, LABELS[field])
        field_scores[field] = score
        weighted += WEIGHTS[field] * score
    return weighted, field_scores


def submission_timeline(label: str) -> str:
    return "more_than_5_years" if label == "longer_than_5_years" else label


def write_submission(path: Path, test_rows: List[dict], pred_rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"],
            lineterminator="\n",
        )
        writer.writeheader()
        for idx, (source, pred) in enumerate(zip(test_rows, pred_rows)):
            writer.writerow({
                "id": source.get("id", idx),
                "promise_status": pred["promise_status"],
                "verification_timeline": submission_timeline(pred["verification_timeline"]),
                "evidence_status": pred["evidence_status"],
                "evidence_quality": pred["evidence_quality"],
            })


def validate_submission(path: Path) -> None:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    expected = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]
    if len(rows) != 2000:
        raise ValueError(f"Expected 2000 rows, got {len(rows)}")
    if list(rows[0].keys()) != expected:
        raise ValueError(f"Unexpected CSV header: {list(rows[0].keys())}")
    if path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise ValueError("CSV has UTF-8 BOM")


def main() -> None:
    set_seed()
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT}. "
            "Place best_model.pt there or set the ESG_CHECKPOINT environment variable. "
            "See README.md for details."
        )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    model_name = args["model_name"]
    max_len = int(args.get("max_len", 256))
    dropout = float(args.get("dropout", 0.1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR if TOKENIZER_DIR.exists() else model_name)
    model = MultiTaskESGModel(model_name, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    biases = load_biases(BIAS_PATH)
    val_rows = load_json(VAL_PATH)
    test_rows = load_json(TEST_PATH)

    val_preds = predict_from_logits(extract_logits(model, tokenizer, val_rows, device, max_len), biases)
    score, field_scores = evaluate(val_rows, val_preds)

    test_preds = predict_from_logits(extract_logits(model, tokenizer, test_rows, device, max_len), biases)
    write_submission(SUBMISSION_PATH, test_rows, test_preds)
    validate_submission(SUBMISSION_PATH)

    print(f"Saved: {SUBMISSION_PATH}")
    print(f"Local validation weighted score: {score:.5f}")
    for field in FIELDS:
        print(f"  {field:<24} {field_scores[field]:.4f}")
    for field in FIELDS:
        print(f"  test {field:<19} {dict(Counter(p[field] for p in test_preds))}")


if __name__ == "__main__":
    main()
