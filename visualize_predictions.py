import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from model import ShadowDetector

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR   = Path("./detection-by-shadow/test_data/test_data")
TRAIN_DIR  = Path("./detection-by-shadow/train_data/train_data")
OUT_DIR    = Path("./viz_output")
OUT_DIR.mkdir(exist_ok=True)

model = ShadowDetector()
model.load_state_dict(torch.load("quick_model.pth", map_location=DEVICE))
model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        bbox_pred, dir_pred = model(t)
    xmin = bbox_pred[0, 0].item() * w
    ymin = bbox_pred[0, 1].item() * h
    xmax = bbox_pred[0, 2].item() * w
    ymax = bbox_pred[0, 3].item() * h
    direction = 1 if dir_pred[0, 0].item() > 0.5 else 0
    return xmin, ymin, xmax, ymax, direction, w, h


def draw_result(img_path, xmin, ymin, xmax, ymax, direction,
                gt_bbox=None, canvas_extra=300):
    """
    Рисуем расширенный canvas: изображение + зона справа/слева
    где отображаем предсказанный bbox человека за кадром
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Расширяем canvas на canvas_extra пикселей с каждой стороны
    total_w = w + 2 * canvas_extra
    canvas = Image.new("RGB", (total_w, h), color=(40, 40, 40))
    canvas.paste(img, (canvas_extra, 0))

    draw = ImageDraw.Draw(canvas)

    # Граница кадра
    draw.rectangle(
        [canvas_extra, 0, canvas_extra + w, h],
        outline=(100, 100, 100), width=2
    )
    draw.text((canvas_extra + 5, 5), "CAMERA FRAME", fill=(150, 150, 150))

    # Предсказанный bbox (со смещением на canvas)
    px1 = xmin + canvas_extra
    py1 = ymin
    px2 = xmax + canvas_extra
    py2 = ymax

    # Pillow expects x1>=x0 and y1>=y0. Predictions can be off-screen and/or inverted.
    x0, x1 = (px1, px2) if px1 <= px2 else (px2, px1)
    y0, y1 = (py1, py2) if py1 <= py2 else (py2, py1)

    # Clamp to canvas to avoid huge coordinates and drawing errors.
    x0 = max(0, min(total_w - 1, x0))
    x1 = max(0, min(total_w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))

    draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
    draw.text((x0, max(0, y0 - 18)), f"PREDICTED dir={'IN' if direction==1 else 'OUT'}",
              fill=(0, 255, 0))

    # Ground truth если есть (только для train данных)
    if gt_bbox:
        gx1, gy1, gx2, gy2 = gt_bbox
        gx1 += canvas_extra
        gx2 += canvas_extra
        gx0, gx1b = (gx1, gx2) if gx1 <= gx2 else (gx2, gx1)
        gy0, gy1b = (gy1, gy2) if gy1 <= gy2 else (gy2, gy1)
        gx0 = max(0, min(total_w - 1, gx0))
        gx1b = max(0, min(total_w - 1, gx1b))
        gy0 = max(0, min(h - 1, gy0))
        gy1b = max(0, min(h - 1, gy1b))
        draw.rectangle([gx0, gy0, gx1b, gy1b], outline=(255, 0, 0), width=3)
        draw.text((gx0, max(0, gy0 - 36)), "GROUND TRUTH", fill=(255, 0, 0))

    # Стрелка показывающая направление движения
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    arrow_len = 60
    arrow_x = cx + (arrow_len if direction == 1 else -arrow_len)
    draw.line([cx, cy, arrow_x, cy], fill=(255, 255, 0), width=4)
    draw.polygon(
        [(arrow_x, cy - 10), (arrow_x, cy + 10),
         (arrow_x + (20 if direction==1 else -20), cy)],
        fill=(255, 255, 0)
    )

    return canvas


# === Визуализируем 5 тестовых + 5 тренировочных (с GT) ===

print("Визуализируем тестовые изображения...")
test_imgs = sorted(TEST_DIR.glob("*.png"))[:5]
for img_path in test_imgs:
    xmin, ymin, xmax, ymax, direction, w, h = predict(img_path)
    print(f"  {img_path.name}: bbox=({xmin:.0f},{ymin:.0f},{xmax:.0f},{ymax:.0f}) dir={direction}")
    canvas = draw_result(img_path, xmin, ymin, xmax, ymax, direction)
    canvas.save(OUT_DIR / f"test_{img_path.stem}.png")

print("\nВизуализируем train (с ground truth для сравнения)...")
train_imgs = sorted(TRAIN_DIR.glob("*.png"))[:5]
for img_path in train_imgs:
    json_path = img_path.with_suffix(".json")
    if not json_path.exists():
        continue

    xmin, ymin, xmax, ymax, direction, w, h = predict(img_path)

    with open(json_path) as f:
        ann = json.load(f)
    tl = ann["bbox"]["top_left"]
    tr = ann["bbox"]["top_right"]
    bl = ann["bbox"]["bottom_left"]
    br = ann["bbox"]["bottom_right"]
    gt = (
        min(tl[0], bl[0]), min(tl[1], tr[1]),
        max(tr[0], br[0]), max(bl[1], br[1])
    )

    canvas = draw_result(img_path, xmin, ymin, xmax, ymax, direction, gt_bbox=gt)
    canvas.save(OUT_DIR / f"train_{img_path.stem}.png")
    print(f"  {img_path.name}: pred=({xmin:.0f},{ymin:.0f}) gt=({gt[0]:.0f},{gt[1]:.0f})")

print(f"\nГотово! Открой папку: {OUT_DIR.absolute()}")