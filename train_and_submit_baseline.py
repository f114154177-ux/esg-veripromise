#!/usr/bin/env python3
"""
ESG_final: supervised multi-task ESG promise verification model.

This is the full training pipeline version, aligned with the baseline notebook:
- Train a shared Chinese Transformer encoder.
- Predict four task labels with four classification heads.
- Evaluate weighted Macro-F1.
- Save the best checkpoint.
- Optionally generate a competition-style submission CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

FIELDS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality",
]

# Internal training label. The official submission label is more_than_5_years.
EVAL_FIELDS = {
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

FIELD_WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}

DROP_COLS = {"esg_type", "company", "ticker", "page_number", "pdf_url", "company_source"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_timeline_label(label: str) -> str:
    if label == "more_than_5_years":
        return "longer_than_5_years"
    return label


def submission_timeline_label(label: str) -> str:
    if label == "longer_than_5_years":
        return "more_than_5_years"
    return label


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}")
    return data


def extract_data_zip(data_zip: Path, extract_dir: Path) -> Optional[Path]:
    if not data_zip.exists():
        return None
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting data zip: {data_zip} -> {extract_dir}")
    with zipfile.ZipFile(data_zip, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def looks_like_test_json(path: Path) -> bool:
    try:
        rows = load_json(path)
    except Exception:
        return False
    if not rows or not isinstance(rows[0], dict):
        return False
    if "data" not in rows[0] or "id" not in rows[0]:
        return False
    # Prefer unlabeled official test data. Some organizers may still include
    # metadata columns; labels should usually be absent.
    label_keys = {"promise_status", "verification_timeline", "evidence_status", "evidence_quality"}
    return not label_keys.issubset(rows[0].keys())


def find_test_json(args: argparse.Namespace) -> Optional[Path]:
    explicit = Path(args.test_path)
    if explicit.exists():
        return explicit

    search_roots = [
        Path(args.output_dir),
        DATA_DIR,
        SCRIPT_DIR,
    ]

    zip_candidates = []
    if getattr(args, "data_zip", None):
        zip_candidates.append(Path(args.data_zip))
    zip_candidates.extend(root / "data.zip" for root in search_roots)

    for zip_path in zip_candidates:
        if zip_path.exists():
            extract_dir = Path(args.output_dir) / "data"
            extract_data_zip(zip_path, extract_dir)
            search_roots.insert(0, extract_dir)
            break

    name_patterns = [
        "*test*.json",
        "*Test*.json",
        "*private*.json",
        "*public*.json",
    ]
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in name_patterns:
            candidates.extend(root.rglob(pattern))

    # Stable, sensible ordering: explicit "test" names first, larger files first.
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            "test" not in p.name.lower(),
            "2000" not in p.name,
            -p.stat().st_size,
            str(p),
        ),
    )
    for path in candidates:
        if looks_like_test_json(path):
            print(f"Auto-detected test JSON: {path}")
            return path
    return None


def clean_and_normalize_rows(rows: Iterable[Dict[str, Any]], has_labels: bool = True) -> List[Dict[str, Any]]:
    cleaned = []
    for row in rows:
        item = {k: v for k, v in row.items() if k not in DROP_COLS}
        if has_labels and "verification_timeline" in item:
            item["verification_timeline"] = normalize_timeline_label(item["verification_timeline"])
        cleaned.append(item)
    return cleaned


def stratified_split_by_promise(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    yes_items = [row for row in rows if row["promise_status"] == "Yes"]
    no_items = [row for row in rows if row["promise_status"] == "No"]

    def split(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        shuffled = items[:]
        random.Random(seed).shuffle(shuffled)
        cut = int(len(shuffled) * (1.0 - val_ratio))
        return shuffled[:cut], shuffled[cut:]

    yes_train, yes_val = split(yes_items)
    no_train, no_val = split(no_items)
    train_data = yes_train + no_train
    val_data = yes_val + no_val
    random.Random(seed).shuffle(train_data)
    random.Random(seed).shuffle(val_data)
    return train_data, val_data


def build_label_maps() -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[int, str]], Dict[str, int]]:
    label2id = {field: {label: idx for idx, label in enumerate(labels)} for field, labels in EVAL_FIELDS.items()}
    id2label = {field: {idx: label for idx, label in enumerate(labels)} for field, labels in EVAL_FIELDS.items()}
    num_labels = {field: len(labels) for field, labels in EVAL_FIELDS.items()}
    return label2id, id2label, num_labels


class ESGDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: Any,
        label2id: Dict[str, Dict[str, int]],
        max_len: int,
        has_labels: bool = True,
    ):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        encoding = self.tokenizer(
            row["data"],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "id": row.get("id", idx),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.has_labels:
            item["labels"] = {
                field: torch.tensor(self.label2id[field][row[field]], dtype=torch.long)
                for field in FIELDS
            }
        return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated = {
        "ids": [item["id"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }
    if "labels" in batch[0]:
        collated["labels"] = {
            field: torch.stack([item["labels"][field] for item in batch])
            for field in FIELDS
        }
    return collated


class MultiTaskESGModel(nn.Module):
    def __init__(self, model_name: str, num_labels: Dict[str, int], dropout: float):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleDict({
            field: nn.Linear(hidden_size, count)
            for field, count in num_labels.items()
        })

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {field: head(pooled) for field, head in self.classifiers.items()}


def compute_class_weights(rows: List[Dict[str, Any]], label2id: Dict[str, Dict[str, int]], device: torch.device) -> Dict[str, torch.Tensor]:
    weights = {}
    for field, mapping in label2id.items():
        counts = Counter(row[field] for row in rows)
        total = sum(counts.values())
        values = []
        for label in mapping:
            count = counts.get(label, 0)
            if count == 0:
                values.append(0.0)
            else:
                values.append(total / (len(mapping) * count))
        weights[field] = torch.tensor(values, dtype=torch.float, device=device)
    return weights


def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return sum(scores) / len(scores)


def evaluate_predictions(gold_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, str]]) -> Tuple[float, Dict[str, float]]:
    field_scores = {}
    weighted = 0.0
    for field, labels in EVAL_FIELDS.items():
        y_true = [row[field] for row in gold_rows]
        y_pred = [row[field] for row in pred_rows]
        score = macro_f1(y_true, y_pred, labels)
        field_scores[field] = score
        weighted += FIELD_WEIGHTS[field] * score
    return weighted, field_scores


def apply_label_rules(pred: Dict[str, str]) -> Dict[str, str]:
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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criteria: Dict[str, nn.Module],
    device: torch.device,
    grad_clip: float,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {field: tensor.to(device) for field, tensor in batch["labels"].items()}

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = sum(criteria[field](logits[field], labels[field]) for field in FIELDS)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if log_every and step % log_every == 0:
            print(f"    step {step:>4}/{len(dataloader)} loss={loss.item():.4f}")

    return total_loss / max(1, len(dataloader))


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    id2label: Dict[str, Dict[int, str]],
    device: torch.device,
    use_rules: bool,
) -> List[Dict[str, str]]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_size = input_ids.size(0)
            for idx in range(batch_size):
                pred = {}
                for field in FIELDS:
                    pred_id = logits[field][idx].argmax().item()
                    pred[field] = id2label[field][pred_id]
                if use_rules:
                    pred = apply_label_rules(pred)
                predictions.append(pred)
    return predictions


def save_checkpoint(
    path: Path,
    model: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    label2id: Dict[str, Dict[str, int]],
    id2label: Dict[str, Dict[int, str]],
    score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "label2id": label2id,
            "id2label": id2label,
            "score": score,
        },
        path,
    )
    tokenizer.save_pretrained(path.parent / "tokenizer")


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def print_label_distribution(name: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n{name} rows: {len(rows)}")
    for field in FIELDS:
        counts = Counter(row[field] for row in rows)
        ordered = {label: counts.get(label, 0) for label in EVAL_FIELDS[field]}
        print(f"  {field:<24} {ordered}")


def write_prediction_json(path: Path, source_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = []
    for source, pred in zip(source_rows, pred_rows):
        item = dict(source)
        item.update(pred)
        output.append(item)
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def write_submission_csv(path: Path, test_rows: List[Dict[str, Any]], pred_rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"],
            lineterminator="\n",
        )
        writer.writeheader()
        for idx, (source, pred) in enumerate(zip(test_rows, pred_rows)):
            writer.writerow(
                {
                    "id": source.get("id", idx),
                    "promise_status": pred["promise_status"],
                    "verification_timeline": submission_timeline_label(pred["verification_timeline"]),
                    "evidence_status": pred["evidence_status"],
                    "evidence_quality": pred["evidence_quality"],
                }
            )


@dataclass
class DataBundle:
    train_rows: List[Dict[str, Any]]
    val_rows: List[Dict[str, Any]]


def load_training_data(args: argparse.Namespace) -> DataBundle:
    if args.train_path and args.val_path:
        train_rows = clean_and_normalize_rows(load_json(Path(args.train_path)), has_labels=True)
        val_rows = clean_and_normalize_rows(load_json(Path(args.val_path)), has_labels=True)
    else:
        rows = clean_and_normalize_rows(load_json(Path(args.all_train_path)), has_labels=True)
        train_rows, val_rows = stratified_split_by_promise(rows, args.val_ratio, args.seed)

    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_val:
        val_rows = val_rows[: args.limit_val]
    return DataBundle(train_rows=train_rows, val_rows=val_rows)


def run_train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    label2id, id2label, num_labels = build_label_maps()

    data = load_training_data(args)
    print_label_distribution("Train", data.train_rows)
    print_label_distribution("Validation", data.val_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = ESGDataset(data.train_rows, tokenizer, label2id, args.max_len, has_labels=True)
    val_dataset = ESGDataset(data.val_rows, tokenizer, label2id, args.max_len, has_labels=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\nDevice: {device}")
    print(f"Model: {args.model_name}")

    model = MultiTaskESGModel(args.model_name, num_labels, args.dropout).to(device)

    class_weights = compute_class_weights(data.train_rows, label2id, device) if args.class_weight else None
    criteria = {}
    for field in FIELDS:
        weight = class_weights[field] if class_weights else None
        criteria[field] = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("\nTraining config")
    print(f"  epochs={args.epochs}")
    print(f"  batch_size={args.batch_size}")
    print(f"  eval_batch_size={args.eval_batch_size}")
    print(f"  max_len={args.max_len}")
    print(f"  lr={args.lr}")
    print(f"  total_steps={total_steps}")
    print(f"  warmup_steps={warmup_steps}")
    print(f"  class_weight={args.class_weight}")
    print(f"  rules={args.rules}")

    best_score = -1.0
    history = []
    best_path = Path(args.output_dir) / "checkpoints" / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criteria,
            device,
            args.grad_clip,
            args.log_every,
        )
        pred_rows = predict(model, val_loader, id2label, device, use_rules=args.rules)
        weighted, field_scores = evaluate_predictions(data.val_rows, pred_rows)
        history.append({"epoch": epoch, "loss": avg_loss, "weighted": weighted, **field_scores})

        print(f"  avg_loss={avg_loss:.4f}")
        print(f"  weighted_macro_f1={weighted:.5f}")
        for field in FIELDS:
            print(f"    {field:<24} macro_f1={field_scores[field]:.4f} weight={FIELD_WEIGHTS[field]}")

        if weighted > best_score:
            best_score = weighted
            save_checkpoint(best_path, model, tokenizer, args, label2id, id2label, best_score)
            print(f"  saved best checkpoint: {best_path} score={best_score:.5f}")

    history_path = Path(args.output_dir) / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\nSaved history: {history_path}")

    print("\nLoading best checkpoint for final validation output...")
    load_checkpoint(best_path, model, device)
    final_preds = predict(model, val_loader, id2label, device, use_rules=args.rules)
    final_weighted, final_field_scores = evaluate_predictions(data.val_rows, final_preds)
    print("\nFinal validation")
    print(f"  weighted_macro_f1={final_weighted:.5f}")
    for field in FIELDS:
        print(f"    {field:<24} macro_f1={final_field_scores[field]:.4f}")

    pred_path = Path(args.output_dir) / "prediction.json"
    write_prediction_json(pred_path, data.val_rows, final_preds)
    print(f"Saved validation predictions: {pred_path}")
    return best_path


def run_submit(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    label2id, id2label, num_labels = build_label_maps()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    test_path = find_test_json(args)
    if test_path is None:
        raise FileNotFoundError(
            "Test data not found. Put vpesg4k_test_2000.json under data/, "
            "or pass --test-path /path/to/test.json"
        )

    test_rows = clean_and_normalize_rows(load_json(test_path), has_labels=False)
    if args.limit_test:
        test_rows = test_rows[: args.limit_test]

    tokenizer_path = checkpoint_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else args.model_name)
    test_dataset = ESGDataset(test_rows, tokenizer, label2id, args.max_len, has_labels=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = MultiTaskESGModel(args.model_name, num_labels, args.dropout).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint validation score: {checkpoint.get('score', 'N/A')}")

    pred_rows = predict(model, test_loader, id2label, device, use_rules=args.rules)
    submission_path = Path(args.submission_path)
    if not submission_path.is_absolute():
        submission_path = Path(args.output_dir) / submission_path
    write_submission_csv(submission_path, test_rows, pred_rows)
    print(f"Saved submission: {submission_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESG_final supervised multi-task training")
    parser.add_argument("--mode", choices=["train", "submit", "train_submit"], default="train")

    parser.add_argument("--model-name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--all-train-path", default=str(DATA_DIR / "vpesg4k_train_1000.json"))
    parser.add_argument("--train-path", default=str(DATA_DIR / "train_800.json"))
    parser.add_argument("--val-path", default=str(DATA_DIR / "val_200.json"))
    parser.add_argument("--test-path", default=str(DATA_DIR / "vpesg4k_test_2000.json"))
    parser.add_argument("--data-zip", default="")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument("--checkpoint", default=str(SCRIPT_DIR / "outputs" / "checkpoints" / "best_model.pt"))
    parser.add_argument("--submission-path", default="submission_baseline.csv")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--class-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rules", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "submit":
        run_submit(args)
    else:
        best_path = run_train(args)
        args.checkpoint = str(best_path)
        detected_test_path = find_test_json(args)
        if detected_test_path is not None:
            args.test_path = str(detected_test_path)
            print("\nTest file found. Generating submission...")
            run_submit(args)
        else:
            print("\nTraining finished, but test file was not found.")
            print(f"Expected test path: {args.test_path}")
            print(f"Or data zip path: {args.data_zip}")
            print("After you place the official test JSON there, run:")
            print(f"  python train_and_submit_baseline.py --mode submit --test-path {args.test_path}")


if __name__ == "__main__":
    main()
