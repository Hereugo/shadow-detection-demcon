"""
Быстрая проверка что всё работает перед запуском обучения.
Запускай: python test_pipeline.py
"""
import torch
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms
from dataset import ShadowDataset
from model import ShadowDetector

DATA_DIR = Path("./detection-by-shadow/train_data/train_data")
if not DATA_DIR.exists():
    DATA_DIR = Path("./data/train_data")


def test_json_parsing():
    """Проверяем что JSON читается правильно"""
    print("\n=== 1. JSON parsing ===")
    
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        print(f"[FAIL] JSON files not found in {DATA_DIR}")
        return False
    
    with open(json_files[0]) as f:
        ann = json.load(f)
    
    print(f"file_name: {ann['file_name']}")
    print(f"walking_into_frame_bool: {ann['walking_into_frame_bool']}")
    
    tl = ann["bbox"]["top_left"]
    br = ann["bbox"]["bottom_right"]
    print(f"top_left: {tl}  (тип: {type(tl)})")
    print(f"bottom_right: {br}")
    
    xs = [ann["bbox"]["top_left"][0], ann["bbox"]["top_right"][0],
          ann["bbox"]["bottom_left"][0], ann["bbox"]["bottom_right"][0]]
    ys = [ann["bbox"]["top_left"][1], ann["bbox"]["top_right"][1],
          ann["bbox"]["bottom_left"][1], ann["bbox"]["bottom_right"][1]]
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    print(f"bbox -> xmin={xmin:.1f} ymin={ymin:.1f} xmax={xmax:.1f} ymax={ymax:.1f}")
    
    # Проверка: координаты должны быть за кадром
    if xmin < 0 or xmax > 720:
        print("[OK] xmin/xmax out of frame (expected).")
    else:
        print("[WARN] Coordinates inside frame - double-check data.")
    
    print("[OK] JSON parsing")
    return True


def test_dataset():
    """Проверяем Dataset класс"""
    print("\n=== 2. Dataset ===")
    
    try:
        ds = ShadowDataset(str(DATA_DIR))
        print(f"Размер датасета: {len(ds)}")
        if len(ds) == 0:
            print("[FAIL] No (image + json) pairs found.")
            print(f"       Expected images next to JSON in {DATA_DIR} (e.g. image_1.png).")
            return False
        
        sample = ds[0]
        print(f"image shape: {sample['image'].shape}")   # должен быть [3, 260, 260]
        print(f"bbox: {sample['bbox']}")                  # 4 нормализованных числа
        print(f"direction: {sample['direction']}")        # 0.0 или 1.0
        print(f"file_name: {sample['file_name']}")
        
        # Проверка нормализации
        bbox = sample['bbox']
        print(f"bbox диапазон: [{bbox.min():.2f}, {bbox.max():.2f}]")
        if bbox.min() < -0.5:
            print("[OK] Negative coordinates present (person off-screen).")
        
        print("[OK] Dataset")
        return True
    except Exception as e:
        print(f"[FAIL] Dataset error: {e}")
        return False


def test_model():
    """Проверяем модель - forward pass"""
    print("\n=== 3. Model forward pass ===")
    
    try:
        model = ShadowDetector()
        model.eval()
        
        # Фейковый батч
        dummy = torch.zeros(2, 3, 260, 260)
        
        with torch.no_grad():
            bbox_pred, dir_pred = model(dummy)
        
        print(f"bbox_pred shape: {bbox_pred.shape}")    # [2, 4]
        print(f"dir_pred shape:  {dir_pred.shape}")     # [2, 1]
        print(f"bbox_pred: {bbox_pred[0]}")
        print(f"dir_pred:  {dir_pred[0]}")
        
        # Проверка что нет sigmoid на bbox (должны быть любые числа)
        if bbox_pred.abs().max() > 2:
            print("[OK] bbox not clamped (good for off-screen).")
        
        print("[OK] Model")
        return True
    except Exception as e:
        print(f"[FAIL] Model error: {e}")
        return False


def test_dataloader():
    """Проверяем DataLoader - batch"""
    print("\n=== 4. DataLoader ===")
    
    try:
        from torch.utils.data import DataLoader
        
        ds = ShadowDataset(str(DATA_DIR))
        if len(ds) == 0:
            print("[FAIL] DataLoader: dataset is empty (no images next to JSON).")
            return False
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        
        batch = next(iter(loader))
        print(f"Batch image shape:  {batch['image'].shape}")    # [4, 3, 260, 260]
        print(f"Batch bbox shape:   {batch['bbox'].shape}")     # [4, 4]
        print(f"Batch direction:    {batch['direction']}")      # [4]
        
        print("[OK] DataLoader")
        return True
    except Exception as e:
        print(f"[FAIL] DataLoader error: {e}")
        return False


def test_one_training_step():
    """Проверяем один шаг обучения"""
    print("\n=== 5. One training step ===")
    
    try:
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        
        ds = ShadowDataset(str(DATA_DIR))
        if len(ds) == 0:
            print("[FAIL] Training step: dataset is empty (no images next to JSON).")
            return False
        loader = DataLoader(ds, batch_size=4, shuffle=True)
        model = ShadowDetector()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        dir_loss_fn = nn.BCELoss()
        
        batch = next(iter(loader))
        images = batch["image"]
        bbox_target = batch["bbox"]
        dir_target = batch["direction"].unsqueeze(1)
        
        optimizer.zero_grad()
        bbox_pred, dir_pred = model(images)
        
        # Простой MSE для проверки (полный loss в train.py)
        loss_bbox = nn.MSELoss()(bbox_pred, bbox_target)
        loss_dir = dir_loss_fn(dir_pred, dir_target)
        loss = loss_bbox + 0.3 * loss_dir
        
        loss.backward()
        optimizer.step()
        
        print(f"loss_bbox: {loss_bbox.item():.4f}")
        print(f"loss_dir:  {loss_dir.item():.4f}")
        print(f"total:     {loss.item():.4f}")
        print("[OK] Training step")
        return True
    except Exception as e:
        print(f"[FAIL] Training step error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("SHADOW DETECTION - PIPELINE TEST")
    print("=" * 50)
    
    results = {
        "JSON parsing":    test_json_parsing(),
        "Dataset":         test_dataset(),
        "Model":           test_model(),
        "DataLoader":      test_dataloader(),
        "Training step":   test_one_training_step(),
    }
    
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ:")
    for name, ok in results.items():
        icon = "[OK]" if ok else "[FAIL]"
        print(f"  {icon} {name}")
    
    all_ok = all(results.values())
    if all_ok:
        print("\n[OK] All checks passed. Run: python train.py")
    else:
        print("\n[FAIL] Fix issues above before training.")