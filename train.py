import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import ShadowDataset
from model import ShadowDetector


def iou_loss(pred, target):
    """
    CIoU loss — лучше чем MSE для bbox регрессии.
    Штрафует за:
    - плохое перекрытие
    - расстояние между центрами  
    - разницу в aspect ratio
    
    Мы используем упрощённый IoU для скорости.
    """
    # Площадь пересечения
    inter_xmin = torch.max(pred[:, 0], target[:, 0])
    inter_ymin = torch.max(pred[:, 1], target[:, 1])
    inter_xmax = torch.min(pred[:, 2], target[:, 2])
    inter_ymax = torch.min(pred[:, 3], target[:, 3])

    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    intersection = inter_w * inter_h

    # Площадь каждого bbox
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

    union = pred_area + target_area - intersection + 1e-6
    iou = intersection / union

    return 1 - iou.mean()  # хотим максимизировать IoU → минимизируем 1-IoU


def train():
    # --- Конфигурация ---
    DATA_DIR = "./data/train_data"
    EPOCHS = 30
    BATCH_SIZE = 16
    LR = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем: {DEVICE}")

    # --- Данные ---
    dataset = ShadowDataset(DATA_DIR)
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Модель ---
    model = ShadowDetector().to(DEVICE)

    # Оптимизатор — AdamW лучше Adam для регрессии
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Scheduler — уменьшает lr по косинусу (плавно)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    dir_loss_fn = nn.BCELoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            bbox_target = batch["bbox"].to(DEVICE)
            dir_target = batch["direction"].to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()

            bbox_pred, dir_pred = model(images)

            # Два loss-а — один за bbox, один за направление
            loss_bbox = iou_loss(bbox_pred, bbox_target)
            loss_dir = dir_loss_fn(dir_pred, dir_target)

            # Суммируем: bbox важнее (вес 1.0 vs 0.3)
            loss = loss_bbox + 0.3 * loss_dir

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                bbox_target = batch["bbox"].to(DEVICE)
                dir_target = batch["direction"].to(DEVICE).unsqueeze(1)

                bbox_pred, dir_pred = model(images)
                loss = iou_loss(bbox_pred, bbox_target) + 0.3 * dir_loss_fn(dir_pred, dir_target)
                val_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | train: {avg_train:.4f} | val: {avg_val:.4f}")

        # Сохраняем лучшую модель
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Saved best model (val_loss={avg_val:.4f})")


if __name__ == "__main__":
    train()