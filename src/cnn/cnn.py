import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, classification_report, auc, roc_curve
from torchvision import models, datasets, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# Model/Training Config
@dataclass
class TrainingConfig:
    """Configuration for the training script."""
    data_dir: Path = Path(r"C:\retinal-ai\uwf_images")
    n_splits: int = 5
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    epochs: int = 30 
    learning_rate: float = 1e-5 
    patience: int = 7 


# Create Model
def create_model():
    """Creates an EfficientNet-B0 model with a custom classifier head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze the first 6 out of 8 feature blocks
    for i in range(6):
        for param in model.features[i].parameters():
            param.requires_grad = False

    # Replace the classifier head
    updated_head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 7)
    )
    model.classifier = updated_head
    return model


# class TransformedSubset(torch.utils.data.Dataset):
#     """
#     A wrapper for a PyTorch Subset that allows applying a specific transform.
#     """
#     def __init__(self, subset, transform=None):
#         self.subset = subset
#         self.transform = transform

#     def __getitem__(self, index):
#         x, y = self.subset[index]
#         if self.transform:
#             x = self.transform(x)
#         return x, y

#     def __len__(self):
#         return len(self.subset)


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.images)

# Apply Data Augmentation
def get_transforms():
    """Defines the training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet Normalization
    ])
    return train_transform, val_transform


def get_dataloaders(all_images, all_labels, train_idx, val_idx, cfg: TrainingConfig):
    """Creates training and validation dataloaders for a given fold."""
    train_transform, val_transform = get_transforms()

    train_imgs = [all_images[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_imgs = [all_images[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    train_set = InMemoryDataset(train_imgs, train_labels, train_transform)
    val_set = InMemoryDataset(val_imgs, val_labels, val_transform)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    curr_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        curr_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return curr_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def build_scheduler(optimizer, epochs, warmup_epochs=3):
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )

# Define Training runs
def run_fold(fold: int, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainingConfig):
    """
    Initializes a model and runs the training and validation for a single fold.
    """
    print(f"\n===== Fold {fold + 1} =====")
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")

    # For each fold, re-initialize the model, optimizer, etc.
    model = create_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)
    scheduler = build_scheduler(optimizer, cfg.epochs)

    best_f1 = -1.0 # Initialize with a value that will always be beaten
    best_model_state = None
    patience_counter = 0

    for epoch in tqdm(range(cfg.epochs), desc=f"Fold {fold+1}", leave=False):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        val_labels, val_preds, _ = evaluate(model, val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        lr = optimizer.param_groups[0]["lr"]

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model_state)
    model.to(DEVICE)
    labels, preds, probs = evaluate(model, val_loader)

    return {"labels": labels, "preds": preds, "probs": probs,
            "best_f1": best_f1, "best_state": best_model_state}


def main():
    cfg = TrainingConfig()
    # Set Seed
    print(f"Using device: {DEVICE}") 
    torch.manual_seed(cfg.seed) 
    np.random.seed(cfg.seed)

    full_dataset = datasets.ImageFolder(cfg.data_dir)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images belonging to {num_classes} classes.")

    print("Preloading images into memory...")
    all_images = []
    all_labels = []
    for i in tqdm(range(len(full_dataset))):
        img, label = full_dataset[i]
        all_images.append(img)
        all_labels.append(label)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    targets = np.array(full_dataset.targets)

    all_fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(targets)), targets)):
        train_loader, val_loader = get_dataloaders(
            full_dataset, train_idx, val_idx, cfg
        )

        fold_results = run_fold(fold, train_loader, val_loader, cfg) # Pass num_classes and cfg
        if fold_results:
            all_fold_results.append(fold_results)

    print("\nCross-validation finished.")
    
    # Display and Analyze Results

    # Concatenate all fold results
    all_labels = np.concatenate([r["labels"] for r in all_fold_results])
    all_preds = np.concatenate([r["preds"] for r in all_fold_results])
    all_probs = np.concatenate([r["probs"] for r in all_fold_results])

    # Classification report
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"F1 (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=full_dataset.classes, zero_division=0))

    # ROC curves
    classes = np.arange(len(full_dataset.classes))
    y_bin = label_binarize(all_labels, classes=classes)

    plt.figure(figsize=(10, 7))
    for i, name in enumerate(full_dataset.classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — EfficientNet-B0 (Cross-Validated)")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("roc_cnn.png", dpi=150)

    # Save best model for further analysis
    best_fold = max(all_fold_results, key=lambda r: r["best_f1"])
    torch.save({
        "model_state_dict": best_fold["best_state"],
        "class_names": full_dataset.classes,
        "num_classes": len(full_dataset.classes),
        "image_size": 224,
    }, "best_model.pth")


if __name__ == "__main__":
    main()