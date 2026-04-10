import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from dataset import ShadowDataset
from model import ShadowDetector
from pathlib import Path

DATA_DIR   = "./detection-by-shadow/train_data/train_data"
EPOCHS     = 15
BATCH_SIZE = 16
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CPU режим — медленно. Установи PyTorch с CUDA!")

dataset = ShadowDataset(DATA_DIR)
val_size   = max(4, int(0.1 * len(dataset)))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

print(f"Train: {train_size} | Val: {val_size}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(DEVICE=="cuda"))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(DEVICE=="cuda"))

model = ShadowDetector().to(DEVICE)
optimizer  = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
dir_loss_fn = nn.BCELoss()

def iou_loss(pred, target):
    ix1 = torch.max(pred[:, 0], target[:, 0])
    iy1 = torch.max(pred[:, 1], target[:, 1])
    ix2 = torch.min(pred[:, 2], target[:, 2])
    iy2 = torch.min(pred[:, 3], target[:, 3])
    inter       = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pred_area   = (pred[:,2]   - pred[:,0])   * (pred[:,3]   - pred[:,1])
    target_area = (target[:,2] - target[:,0]) * (target[:,3] - target[:,1])
    union = pred_area + target_area - inter + 1e-6
    return (1 - (inter / union).clamp(0, 1)).mean()

def calc_mean_iou(loader):
    model.eval()
    ious = []
    with torch.no_grad():
        for batch in loader:
            bp = model(batch["image"].to(DEVICE))[0]
            bt = batch["bbox"].to(DEVICE)
            ix1 = torch.max(bp[:,0], bt[:,0])
            iy1 = torch.max(bp[:,1], bt[:,1])
            ix2 = torch.min(bp[:,2], bt[:,2])
            iy2 = torch.min(bp[:,3], bt[:,3])
            inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
            pa = (bp[:,2]-bp[:,0]) * (bp[:,3]-bp[:,1])
            ta = (bt[:,2]-bt[:,0]) * (bt[:,3]-bt[:,1])
            iou = (inter / (pa + ta - inter + 1e-6)).clamp(0, 1)
            ious.extend(iou.cpu().tolist())
    return sum(ious) / len(ious) if ious else 0.0

best_iou = 0.0

print("\nНачинаем обучение...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches  = len(train_loader)

    for i, batch in enumerate(train_loader):
        images   = batch["image"].to(DEVICE)
        bbox_tgt = batch["bbox"].to(DEVICE)
        dir_tgt  = batch["direction"].to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        bbox_pred, dir_pred = model(images)

        loss_bbox = iou_loss(bbox_pred, bbox_tgt)
        loss_dir  = dir_loss_fn(dir_pred, dir_tgt)
        loss      = loss_bbox + 0.3 * loss_dir

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Прогресс внутри эпохи
        pct  = (i + 1) / n_batches * 100
        bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  Epoch {epoch+1:02d}/{EPOCHS} [{bar}] {pct:5.1f}%  loss={loss.item():.4f}", end="", flush=True)

    avg_loss = total_loss / n_batches
    val_iou  = calc_mean_iou(val_loader)

    print(f"\r  Epoch {epoch+1:02d}/{EPOCHS} ✓  avg_loss={avg_loss:.4f}  val_IoU={val_iou:.3f}   ")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "best_model.pth")
        print(f"    → Saved best_model.pth (IoU={best_iou:.3f})")

# Финальный save в любом случае
torch.save(model.state_dict(), "quick_model.pth")
print(f"\nГотово! best_model.pth (IoU={best_iou:.3f}) | quick_model.pth (последний epoch)")
print("Следующий шаг: python visualize_predictions.py")