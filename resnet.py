"""
ResNet-50 on PlantVillage dataset — train, evaluate, save metrics.
Uses torchvision's ResNet-50 (trained from scratch).
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
from torchvision import datasets, models, transforms

# ──────────────────────────── Config ────────────────────────────
DATA_DIR = "/homes/ndavoli/dataset/PlantVillage-color"
WEIGHTS_DIR = "/homes/ndavoli/resnet/weights"
METRICS_PATH = "/homes/ndavoli/resnet/metrics.json"

BATCH_SIZE = 128
NUM_EPOCHS = 30
LR = 0.1              # ResNet default — higher LR works well with BN
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4    # ResNet paper default
NUM_WORKERS = 4
TEST_SPLIT = 0.2
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────── Data ──────────────────────────────
# ResNet-50 expects 224x224 input
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

# Apply different transforms to train/test via wrapper
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
# ResNet-50 from scratch, adjust final FC layer to our number of classes
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                            momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# Cosine annealing scheduler for better convergence
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ──────────────────────────── Train ─────────────────────────────
scaler = torch.amp.GradScaler("cuda")  # mixed precision for speed

train_start = time.time()
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
weights_path = os.path.join(WEIGHTS_DIR, "resnet50.pth")
torch.save(model.state_dict(), weights_path)
model_size_bytes = os.path.getsize(weights_path)
num_params = sum(p.numel() for p in model.parameters())

# ──────────────────────────── Test ──────────────────────────────
model.eval()
all_labels = []
all_preds = []
all_probs = []

inference_start = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)  # full precision — avoids fp16 overflow in logits
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(outputs.argmax(1).cpu().numpy())
        all_labels.append(labels.numpy())

inference_time = time.time() - inference_start

all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)

# ──────────────────────────── Metrics ───────────────────────────
# Top-1 accuracy
top1_acc = (all_preds == all_labels).mean()

# Top-5 accuracy
top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
top5_acc = np.mean([all_labels[i] in top5_preds[i] for i in range(len(all_labels))])

# Balanced accuracy
balanced_acc = balanced_accuracy_score(all_labels, all_preds)

# Precision, recall, F1 (macro-averaged)
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

# Predictive entropy: -sum(p * log(p)) averaged over test samples
entropy = -np.sum(all_probs * np.log(all_probs + 1e-12), axis=1)
mean_entropy = float(entropy.mean())

# ROC curve (one-vs-rest per class)
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
