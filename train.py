import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import ShadowDataset
from model import ShadowDetector


CONFIG = {
    "json_dir":       "./train_data_with_shadow_bbox/train_A",
    "img_dir":        "./detection-by-shadow/train_data/train_data",
    "epochs":         60,
    "batch_size":     16,
    "lr":             2e-4,
    "val_split":      0.15,
    "save_path":     "best_model.pth",
    "warmup_epochs":  15,
    "loss_weight_bbox": 1.0,
    "loss_weight_dir":  0.5,
}


def iou_loss(pred, target):
    ix1 = torch.max(pred[:, 0], target[:, 0])
    iy1 = torch.max(pred[:, 1], target[:, 1])
    ix2 = torch.min(pred[:, 2], target[:, 2])
    iy2 = torch.min(pred[:, 3], target[:, 3])
    inter       = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pred_area   = (pred[:,2] - pred[:,0]) * (pred[:,3] - pred[:,1])
    target_area = (target[:,2] - target[:,0]) * (target[:,3] - target[:,1])
    union = pred_area + target_area - inter + 1e-6
    return (1 - (inter / union).clamp(0, 1)).mean()


def combined_bbox_loss(pred, target, epoch: int):
    """
    Warmup фаза (epoch < warmup_epochs): только MSE
      - MSE даёт градиент даже когда боксы не пересекаются
      - Модель учится двигаться в правильный диапазон координат

    После warmup: MSE + IoU плавно
      - IoU финально оптимизирует перекрытие боксов
    """
    mse = nn.MSELoss()(pred, target)
    iou = iou_loss(pred, target)

    if epoch < CONFIG["warmup_epochs"]:
        return mse, float(mse.item()), 0.0
    else:
        # Плавный переход от MSE к IoU за 10 эпох
        alpha = min(1.0, (epoch - CONFIG["warmup_epochs"]) / 10.0)
        loss  = (1 - alpha) * mse + alpha * iou
        return loss, float(mse.item()), float(iou.item())


def compute_metrics(loader, model, device):
    model.eval()
    ious, dir_correct, kaggle_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            images  = batch["image"].to(device)
            sf      = batch["shadow_features"].to(device)
            bt      = batch["bbox"].to(device)
            dir_tgt = batch["direction"].to(device)

            bp, dp = model(images, sf)

            ix1 = torch.max(bp[:,0], bt[:,0])
            iy1 = torch.max(bp[:,1], bt[:,1])
            ix2 = torch.min(bp[:,2], bt[:,2])
            iy2 = torch.min(bp[:,3], bt[:,3])
            inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
            pa  = (bp[:,2] - bp[:,0]) * (bp[:,3] - bp[:,1])
            ta  = (bt[:,2] - bt[:,0]) * (bt[:,3] - bt[:,1])
            iou = (inter / (pa + ta - inter + 1e-6)).clamp(0, 1)

            dir_binary = (dp.squeeze(1) > 0.5).float()
            correct    = (dir_binary == dir_tgt).float()
            bonus      = torch.where(
                correct.bool(),
                torch.full_like(correct,  0.05),
                torch.full_like(correct, -0.05)
            )

            ious.extend(iou.cpu().tolist())
            dir_correct.extend(correct.cpu().tolist())
            kaggle_scores.extend((iou + bonus).cpu().tolist())

    return {
        "mean_iou":     sum(ious) / len(ious),
        "dir_acc":      sum(dir_correct) / len(dir_correct),
        "kaggle_score": sum(kaggle_scores) / len(kaggle_scores),
    }


def train():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Данные ---
    dataset  = ShadowDataset(CONFIG["json_dir"], CONFIG["img_dir"])
    val_size = max(4, int(CONFIG["val_split"] * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=0
    )

    # --- Модель ---
    model       = ShadowDetector().to(DEVICE)
    optimizer   = AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler   = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    dir_loss_fn = nn.BCELoss()
    best_kaggle = -999.0

    print(f"\nОбучение {CONFIG['epochs']} эпох "
          f"(warmup MSE: {CONFIG['warmup_epochs']} эпох, потом IoU)\n")

    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        n = len(train_loader)

        for i, batch in enumerate(train_loader):
            images  = batch["image"].to(DEVICE)
            sf      = batch["shadow_features"].to(DEVICE)
            bt      = batch["bbox"].to(DEVICE)
            dir_tgt = batch["direction"].to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            bp, dp = model(images, sf)

            loss_bbox, mse_val, iou_val = combined_bbox_loss(bp, bt, epoch)
            loss_dir  = dir_loss_fn(dp, dir_tgt)
            loss = (CONFIG["loss_weight_bbox"] * loss_bbox
                  + CONFIG["loss_weight_dir"]  * loss_dir)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pct = (i + 1) / n * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            phase = "MSE" if epoch < CONFIG["warmup_epochs"] else "IoU"
            print(
                f"\r  Epoch {epoch+1:02d}/{CONFIG['epochs']} [{bar}] {pct:4.0f}%"
                f"  loss={loss.item():.4f}  mse={mse_val:.3f}  iou={iou_val:.3f}  [{phase}]",
                end="", flush=True
            )

        scheduler.step()

        metrics = compute_metrics(val_loader, model, DEVICE)
        print(
            f"\r  Epoch {epoch+1:02d}/{CONFIG['epochs']} ✓  "
            f"loss={total_loss/n:.4f} | "
            f"IoU={metrics['mean_iou']:.3f} | "
            f"dir={metrics['dir_acc']:.2f} | "
            f"kaggle={metrics['kaggle_score']:.3f}   "
        )

        if metrics["kaggle_score"] > best_kaggle:
            best_kaggle = metrics["kaggle_score"]
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"    → Saved {CONFIG['save_path']} (kaggle={best_kaggle:.3f})")

    print(f"\nГотово! Best kaggle={best_kaggle:.3f}")
    print("Следующий шаг: python predict.py")


if __name__ == "__main__":
    train()