"""
Neural Network Classifier for Stringified Image Dataset
3-class classification (labels: 0, 1, 2)

Usage:
    python train.py [--data_dir PATH] [--epochs N] [--optimizer {adam,adamw,sgd}]
                    [--student_id ID] [--batch_size N] [--lr LR]
                    [--backbone {efficientnet_b0,resnet50,vit_b_16}]
"""

import argparse
import os
import random
import tarfile
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Step 1 – Data Pipeline
# ---------------------------------------------------------------------------

class StringImageDataset(Dataset):
    """Custom Dataset for the stringified image classification task."""

    def __init__(self, image_dir: str, labels: pd.Series | None, image_ids: list[str],
                 transform=None):
        """
        Args:
            image_dir:  Folder that contains the image files.
            labels:     Pandas Series (index = filename, value = int label).
                        Pass None for the test set where labels are unknown.
            image_ids:  Ordered list of image filenames to include.
            transform:  torchvision transforms to apply.
        """
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        fname = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, fname)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = int(self.labels[fname])
            return image, label
        return image, fname   # test set: return filename instead of label


def build_transforms(image_size: int = 224, is_train: bool = True):
    """Build torchvision transform pipelines for train / val / test."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def load_data(data_dir: str, val_fraction: float = 0.2, image_size: int = 224,
              batch_size: int = 32, num_workers: int = 4, seed: int = 42):
    """
    Extract the dataset archive if needed, then build DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, label_series
    """
    # ---- locate / extract archive ----------------------------------------
    archive = os.path.join(data_dir, "string_dataset_3_classes_cleaned.tar.gz")
    extracted_root = os.path.join(data_dir, "string_dataset_3_classes_cleaned")

    if not os.path.isdir(extracted_root):
        print(f"Extracting dataset to {data_dir} …")
        with tarfile.open(archive, "r:gz") as tf:
            # Skip macOS metadata entries (._*)
            members = [m for m in tf.getmembers()
                       if not os.path.basename(m.name).startswith("._")]
            tf.extractall(path=data_dir, members=members)
        print("Extraction done.")

    train_img_dir = os.path.join(extracted_root, "train")
    test_img_dir  = os.path.join(extracted_root, "test")
    csv_path      = os.path.join(extracted_root, "train.csv")

    # ---- read annotations ------------------------------------------------
    df = pd.read_csv(csv_path)
    df["Label"] = df["Label"].astype(int)
    label_series = df.set_index("ID")["Label"]

    all_ids = df["ID"].tolist()
    all_labels = df["Label"].tolist()

    # ---- stratified train / val split ------------------------------------
    train_ids, val_ids = train_test_split(
        all_ids,
        test_size=val_fraction,
        stratify=all_labels,
        random_state=seed,
    )
    print(f"Train samples: {len(train_ids)}  |  Val samples: {len(val_ids)}")

    # ---- test images (no labels) -----------------------------------------
    test_ids = sorted(
        f for f in os.listdir(test_img_dir)
        if f.endswith(".JPEG") and not f.startswith("._")
    )
    print(f"Test samples : {len(test_ids)}")

    # ---- datasets --------------------------------------------------------
    train_ds = StringImageDataset(
        train_img_dir, label_series, train_ids,
        transform=build_transforms(image_size, is_train=True))

    val_ds = StringImageDataset(
        train_img_dir, label_series, val_ids,
        transform=build_transforms(image_size, is_train=False))

    test_ds = StringImageDataset(
        test_img_dir, None, test_ids,
        transform=build_transforms(image_size, is_train=False))

    # ---- loaders ---------------------------------------------------------
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, label_series


# ---------------------------------------------------------------------------
# Step 2 – Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 3, backbone: str = "efficientnet_b0",
                pretrained: bool = True) -> nn.Module:
    """
    Build a fine-tunable classification model via torchvision.

    Supported backbone strings:
        "efficientnet_b0", "resnet50", "vit_b_16"
    """
    if backbone == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "resnet50":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone == "vit_b_16":
        weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. "
                         "Choose from: efficientnet_b0, resnet50, vit_b_16")
    return model


def build_optimizer(model: nn.Module, optimizer_name: str = "adamw",
                    lr: float = 1e-4, weight_decay: float = 1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    name = optimizer_name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9,
                         weight_decay=weight_decay, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}. "
                         "Choose from: adam, adamw, sgd")


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss, correct, n = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += len(labels)

    return total_loss / n, correct / n


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Backbone   : {args.backbone}")
    print(f"Optimizer  : {args.optimizer}  |  lr={args.lr}  |  epochs={args.epochs}")

    # ---- data ----------------------------------------------------------------
    train_loader, val_loader, test_loader, _ = load_data(
        data_dir    = args.data_dir,
        val_fraction= args.val_fraction,
        image_size  = args.image_size,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        seed        = args.seed,
    )

    # ---- model ---------------------------------------------------------------
    model = build_model(num_classes=3, backbone=args.backbone,
                        pretrained=args.pretrained).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    # Cosine annealing lr schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- training loop -------------------------------------------------------
    best_val_acc = 0.0
    weights_path = os.path.join(args.data_dir, "best_model.pth")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device, is_train=True)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion,
                                          optimizer, device, is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        history.append(dict(epoch=epoch, train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss,
                            val_acc=val_acc))

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), weights_path)
            print(f"  → New best val acc: {best_val_acc:.4f}  (saved to {weights_path})")

    print(f"\nTraining complete.  Best val acc: {best_val_acc:.4f}")

    # ---- save training log ---------------------------------------------------
    log_path = os.path.join(args.data_dir, "training_log.csv")
    pd.DataFrame(history).to_csv(log_path, index=False)
    print(f"Training log saved to {log_path}")

    # ---- Step 3: generate test predictions -----------------------------------
    predict(model, weights_path, test_loader, args)


# ---------------------------------------------------------------------------
# Step 3 – Prediction
# ---------------------------------------------------------------------------

def predict(model, weights_path: str, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    image_ids, preds = [], []
    with torch.no_grad():
        for images, fnames in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            batch_preds = logits.argmax(1).cpu().tolist()
            preds.extend(batch_preds)
            image_ids.extend(fnames)

    out_df = pd.DataFrame({"ID": image_ids, "Label": preds})
    # Sort by ID for deterministic output
    out_df = out_df.sort_values("ID").reset_index(drop=True)

    out_path = os.path.join(args.data_dir, f"{args.student_id}_predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}  ({len(out_df)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stringified-image classifier")

    p.add_argument("--data_dir",    default="/home/runner/work/-/-",
                   help="Root directory containing the .tar.gz archive")
    p.add_argument("--student_id",  default="studentID",
                   help="Your student ID (used for output filename)")
    p.add_argument("--backbone",    default="efficientnet_b0",
                   choices=["efficientnet_b0", "resnet50", "vit_b_16"],
                   help="Backbone architecture (default: efficientnet_b0)")
    p.add_argument("--pretrained",  action="store_true", default=True,
                   help="Use ImageNet-pretrained weights (default: True)")
    p.add_argument("--optimizer",   default="adamw",
                   choices=["adam", "adamw", "sgd"])
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--image_size",  type=int,   default=224)
    p.add_argument("--val_fraction",type=float, default=0.2)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
