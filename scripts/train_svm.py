"""Train a classical SVM baseline using HOG + color histograms.

Gracefully exits when data is missing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.hog_color import extract_hog_color  # noqa: E402


def load_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        print(f"labels.csv not found at {labels_path}; skipping SVM training.")
        return pd.DataFrame()
    df = pd.read_csv(labels_path)
    if "image_path" not in df.columns or "class" not in df.columns:
        raise ValueError("labels.csv must have columns: image_path,class")
    return df[["image_path", "class"]]


def compute_features(df: pd.DataFrame, root: Path, image_size: int, max_samples: int | None):
    rows = df.to_dict("records")
    if max_samples:
        rows = rows[:max_samples]
    X = []
    y = []
    for row in rows:
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = root / row["image_path"]
        feat = extract_hog_color(path, image_size=image_size)
        X.append(feat)
        y.append(row["class"])
    return np.stack(X), np.array(y)


def train_and_eval(X: np.ndarray, y: np.ndarray, n_splits: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        clf = LinearSVC(class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[va])
        metrics.append(
            {
                "fold": fold,
                "acc": accuracy_score(y[va], pred),
                "bacc": balanced_accuracy_score(y[va], pred),
                "f1_macro": f1_score(y[va], pred, average="macro"),
            }
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("data/mock/labels.csv"), help="Path to labels.csv")
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root if image_path is relative")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--out", type=Path, default=Path("reports/svm_metrics.csv"))
    args = parser.parse_args()

    df = load_labels(args.labels)
    if df.empty:
        return 0

    X, y = compute_features(df, args.root, args.image_size, args.max_samples)
    metrics = train_and_eval(X, y, args.n_splits)

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.out, index=False)

    print("Fold metrics:")
    for m in metrics:
        print(f"fold={m['fold']} acc={m['acc']:.3f} bacc={m['bacc']:.3f} f1={m['f1_macro']:.3f}")
    print(f"Saved metrics to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
