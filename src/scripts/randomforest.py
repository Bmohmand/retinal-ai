import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, accuracy_score, f1_score, roc_curve, auc)
from sklearn.pipeline import Pipeline


def load_features(dir):
    
    feature_dir = Path(dir)

    # Color features
    df = pd.read_csv(feature_dir / "features.csv")
    y = df["label"].values
    X_color = df.drop(columns=["filename", "label"]).values

    # HOG features
    hog_df = np.load(feature_dir / "hog_features.npz")
    X_hog = hog_df["features"]

    # Combine and scale
    X = np.hstack((X_color, X_hog))

    return X, y


def train_model(X, y):

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    rf_classifier = RandomForestClassifier(n_estimators=300, 
                                           class_weight="balanced", 
                                           max_depth=30, 
                                           random_state=42,
                                           n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = np.empty(len(y), dtype=y.dtype)
    y_score = np.zeros((len(y), len(np.unique(y))))
    for train_idx, test_idx in tqdm(cv.split(X, y), total=5, desc="Cross-validating"):
        pipeline = Pipeline([
            ("scaler", scaler),
            ("rf", rf_classifier)
        ])
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = pipeline.predict(X[test_idx])
        y_score[test_idx] = pipeline.predict_proba(X[test_idx])


    # Report data
    accuracy = accuracy_score(y, y_pred)

    print(f"Accuracy: {accuracy}")
    print("\n Classification Report")
    print(classification_report(y, y_pred, zero_division=0))

    # ROC AUC Plotting
    plt.figure(figsize=(10, 7))

    # Loop through each class to calculate and plot ROC/AUC (One-vs-Rest)
    classes = np.unique(y)
    for i, class_label in enumerate(classes):
        # Binarize labels for the current class
        y_bin = (y == class_label).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RandomForest ROC Curves for Retinal Classification - Fundus')
    plt.legend(loc='lower right')
    plt.savefig("randomforest.png")


if __name__ == "__main__":
    features_dir = r"e:\retinal-ai\imbalanced-2031\imbalanced_fundus_features"

    X, y = load_features(features_dir)
    print(f"Loaded Data {X.shape[0]} images, {X.shape[1]} features\n")
    train_model(X, y)