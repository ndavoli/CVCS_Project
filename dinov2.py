"""
DINOv2 (ViT-B/14) on PlantVillage dataset — train linear head, evaluate, save metrics.
Uses the pretrained DINOv2 backbone as a frozen feature extractor with a linear classifier.
DINOv2 is a self-supervised Vision Transformer; training from scratch is not feasible
without massive compute, so the standard evaluation protocol is linear probing.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ──────────────────────────── Config ────────────────────────────
DATA_DIR = "/homes/ndavoli/dataset/PlantVillage-color"
WEIGHTS_DIR = "/homes/ndavoli/dinov2/weights"
METRICS_PATH = "/homes/ndavoli/dinov2/metrics.json"

BATCH_SIZE = 128
NUM_EPOCHS = 30
LR = 0.001             # lower LR for linear probing
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0     # no regularization on a single linear layer
NUM_WORKERS = 4
TEST_SPLIT = 0.2
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────── Data ──────────────────────────────
# DINOv2 ViT-B/14 expects 224x224 input (patch size 14, 16x16 patches)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load full dataset (folder structure: class_name/image.jpg)
full_dataset = datasets.ImageFolder(DATA_DIR)
num_classes = len(full_dataset.classes)

# Deterministic train/test split (same seed for fair comparison)
generator = torch.Generator().manual_seed(SEED)
test_size = int(len(full_dataset) * TEST_SPLIT)
train_size = len(full_dataset) - test_size
train_subset, test_subset = torch.utils.data.random_split(
    full_dataset, [train_size, test_size], generator=generator
)

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)

train_set = TransformSubset(train_subset, train_transform)
test_set = TransformSubset(test_subset, test_transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# ──────────────────────────── Model ─────────────────────────────
# Load pretrained DINOv2 ViT-B/14 backbone
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False  # freeze backbone

embed_dim = backbone.embed_dim  # 768 for ViT-B

# Linear classifier on top of frozen features
classifier = nn.Linear(embed_dim, num_classes).to(device)
backbone = backbone.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=LR, momentum=MOMENTUM,
                            weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ──────────────────────────── Train ─────────────────────────────
# Only the linear head is trained; backbone is frozen
train_start = time.time()
for epoch in range(NUM_EPOCHS):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Extract features (no grad needed for backbone)
        with torch.no_grad():
            features = backbone(images)

        outputs = classifier(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

train_time = time.time() - train_start

# ──────────────────────────── Save weights ──────────────────────
os.makedirs(WEIGHTS_DIR, exist_ok=True)
# Save only the linear head (backbone is pretrained and publicly available)
weights_path = os.path.join(WEIGHTS_DIR, "dinov2_head.pth")
torch.save(classifier.state_dict(), weights_path)
head_size_bytes = os.path.getsize(weights_path)

# Report full model size (backbone + head) for fair comparison
backbone_params = sum(p.numel() for p in backbone.parameters())
head_params = sum(p.numel() for p in classifier.parameters())
num_params = backbone_params + head_params
# Estimate full model size: 4 bytes per float32 parameter + overhead
model_size_bytes = head_size_bytes  # only head is saved

# ──────────────────────────── Test ──────────────────────────────
classifier.eval()
all_labels = []
all_preds = []
all_probs = []

inference_start = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        features = backbone(images)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(outputs.argmax(1).cpu().numpy())
        all_labels.append(labels.numpy())

inference_time = time.time() - inference_start

all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)

# ──────────────────────────── Metrics ───────────────────────────
top1_acc = (all_preds == all_labels).mean()

top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
top5_acc = np.mean([all_labels[i] in top5_preds[i] for i in range(len(all_labels))])

balanced_acc = balanced_accuracy_score(all_labels, all_preds)

precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

entropy = -np.sum(all_probs * np.log(all_probs + 1e-12), axis=1)
mean_entropy = float(entropy.mean())

roc_data = {}
for c in range(num_classes):
    binary_labels = (all_labels == c).astype(int)
    fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, c])
    roc_auc = auc(fpr, tpr)
    roc_data[full_dataset.classes[c]] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": roc_auc,
    }

# ──────────────────────────── Save metrics ──────────────────────
metrics = {
    "top1_accuracy": float(top1_acc),
    "top5_accuracy": float(top5_acc),
    "balanced_accuracy": float(balanced_acc),
    "precision_macro": float(precision),
    "recall_macro": float(recall),
    "f1_macro": float(f1),
    "predictive_entropy_mean": mean_entropy,
    "model_size_bytes": model_size_bytes,
    "num_parameters": num_params,
    "train_time_seconds": round(train_time, 2),
    "inference_time_seconds": round(inference_time, 2),
    "roc_curve": roc_data,
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nDone. Metrics saved to {METRICS_PATH}")
print(f"Top-1: {top1_acc:.4f}  Top-5: {top5_acc:.4f}  F1: {f1:.4f}")
print(f"Train: {train_time:.1f}s  Inference: {inference_time:.1f}s")
print(f"Backbone params: {backbone_params/1e6:.1f}M  Head params: {head_params/1e6:.3f}M  Total: {num_params/1e6:.1f}M")
