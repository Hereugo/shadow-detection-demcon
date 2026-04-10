import json
import pandas as pd
import cv2
import numpy as np
from pathlib import Path

SUBMISSION_CSV = "./submission.csv"
TEST_IMG_DIR   = Path("./detection-by-shadow/test_data/test_data")
TRAIN_IMG_DIR  = Path("./detection-by-shadow/train_data/train_data")
TRAIN_JSON_DIR = Path("./train_data_with_shadow_bbox/train_A_test")
VIZ_DIR        = Path("./viz_results")
VIZ_DIR.mkdir(exist_ok=True)

CANVAS_PAD = 300  # пикселей добавляем с каждой стороны


def draw_prediction(img_path: Path, pred: dict, gt: dict = None) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    img_h, img_w = img.shape[:2]
    total_w = img_w + 2 * CANVAS_PAD

    # Серый расширенный canvas
    canvas = np.full((img_h, total_w, 3), (45, 45, 45), dtype=np.uint8)
    canvas[:, CANVAS_PAD:CANVAS_PAD + img_w] = img

    # Граница кадра
    cv2.rectangle(canvas, (CANVAS_PAD, 0), (CANVAS_PAD + img_w - 1, img_h - 1),
                  (100, 100, 100), 2)
    cv2.putText(canvas, "CAMERA FRAME", (CANVAS_PAD + 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

    def to_canvas_x(x): return int(x) + CANVAS_PAD
    def clamp_rect(x1, y1, x2, y2):
        return (max(0, min(x1, total_w-1)),
                max(0, min(y1, img_h-1)),
                max(0, min(x2, total_w-1)),
                max(0, min(y2, img_h-1)))

    # Ground truth — красный (только для train)
    if gt:
        gx1 = to_canvas_x(gt["xmin"])
        gy1 = int(gt["ymin"])
        gx2 = to_canvas_x(gt["xmax"])
        gy2 = int(gt["ymax"])
        rx1, ry1, rx2, ry2 = clamp_rect(gx1, gy1, gx2, gy2)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 0, 220), 2)
        cv2.putText(canvas, "GT", (rx1, max(ry1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)

    # Предсказание — зелёный
    px1 = to_canvas_x(pred["xmin"])
    py1 = int(pred["ymin"])
    px2 = to_canvas_x(pred["xmax"])
    py2 = int(pred["ymax"])
    cx1, cy1, cx2, cy2 = clamp_rect(px1, py1, px2, py2)
    cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 220, 80), 2)
    cv2.putText(canvas, f"PRED dir={'IN' if pred['direction']==1 else 'OUT'}",
                (cx1, max(cy1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)

    # Стрелка направления из центра predicted bbox
    pcx = (cx1 + cx2) // 2
    pcy = (cy1 + cy2) // 2
    arrow_dx = 70 if pred["direction"] == 1 else -70
    arrow_tx = np.clip(pcx + arrow_dx, 0, total_w - 1)
    cv2.arrowedLine(canvas, (pcx, pcy), (arrow_tx, pcy),
                    (0, 220, 220), 3, tipLength=0.3)

    # IoU если есть GT
    if gt:
        ix1_ = max(pred["xmin"], gt["xmin"])
        iy1_ = max(pred["ymin"], gt["ymin"])
        ix2_ = min(pred["xmax"], gt["xmax"])
        iy2_ = min(pred["ymax"], gt["ymax"])
        inter = max(0, ix2_ - ix1_) * max(0, iy2_ - iy1_)
        pa = (pred["xmax"] - pred["xmin"]) * (pred["ymax"] - pred["ymin"])
        ta = (gt["xmax"]   - gt["xmin"])   * (gt["ymax"]   - gt["ymin"])
        iou = inter / (pa + ta - inter + 1e-6)
        iou = max(0, min(1, iou))

        color = (0, 200, 0) if iou > 0.5 else (0, 165, 255) if iou > 0.3 else (0, 0, 220)
        cv2.putText(canvas, f"IoU={iou:.3f}", (10, img_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Имя файла
    cv2.putText(canvas, pred["id"], (CANVAS_PAD + 8, img_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return canvas


def visualize_test(n=10):
    """Визуализация тестовых изображений (без GT)"""
    df = pd.read_csv(SUBMISSION_CSV)
    out = VIZ_DIR / "test"
    out.mkdir(exist_ok=True)

    print(f"\nВизуализируем {n} тестовых изображений → {out}/")
    for _, row in df.head(n).iterrows():
        img_path = TEST_IMG_DIR / f"{row['id']}.png"
        if not img_path.exists():
            continue

        pred = dict(row)
        canvas = draw_prediction(img_path, pred)
        if canvas is not None:
            cv2.imwrite(str(out / f"{row['id']}.png"), canvas)
            print(f"  {row['id']}: bbox=({row['xmin']:.0f},{row['ymin']:.0f},"
                  f"{row['xmax']:.0f},{row['ymax']:.0f}) dir={row['direction']}")


def visualize_train(n=20):
    """
    Визуализация train изображений — сравниваем prediction vs GT.
    Запускаем модель на train чтобы увидеть насколько хорошо учится.
    """
    import torch
    from torchvision import transforms
    from PIL import Image
    from model import ShadowDetector
    from dataset import ShadowDataset

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = ShadowDetector()
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval().to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    out = VIZ_DIR / "train"
    out.mkdir(exist_ok=True)

    json_files = sorted(TRAIN_JSON_DIR.glob("*.json"))[:n]
    ious = []

    print(f"\nВизуализируем {len(json_files)} train изображений → {out}/")

    for json_path in json_files:
        img_path = TRAIN_IMG_DIR / f"{json_path.stem}.png"
        if not img_path.exists():
            continue

        with open(json_path) as f:
            ann = json.load(f)

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Shadow features
        sf = torch.zeros(10)
        if "shadow_bbox" in ann and ann["shadow_bbox"] and "xyxy" in ann["shadow_bbox"]:
            s = ann["shadow_bbox"]["xyxy"]
            s_xmin = s["xmin"] / orig_w
            s_ymin = s["ymin"] / orig_h
            s_xmax = s["xmax"] / orig_w
            s_ymax = s["ymax"] / orig_h
            sf = torch.tensor([
                s_xmin, s_ymin, s_xmax, s_ymax,
                (s_xmin+s_xmax)/2, (s_ymin+s_ymax)/2,
                s_xmax-s_xmin, s_ymax-s_ymin,
                0.0, 0.0
            ])

        img_t = transform(img).unsqueeze(0).to(DEVICE)
        sf_t  = sf.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            bp, dp = model(img_t, sf_t)

        pred = {
            "id":        json_path.stem,
            "xmin":      bp[0,0].item() * orig_w,
            "ymin":      bp[0,1].item() * orig_h,
            "xmax":      bp[0,2].item() * orig_w,
            "ymax":      bp[0,3].item() * orig_h,
            "direction": 1 if dp[0,0].item() > 0.5 else 0,
        }

        tl = ann["bbox"]["top_left"]
        tr = ann["bbox"]["top_right"]
        bl = ann["bbox"]["bottom_left"]
        br = ann["bbox"]["bottom_right"]
        gt = {
            "xmin": min(tl[0], bl[0]),
            "xmax": max(tr[0], br[0]),
            "ymin": min(tl[1], tr[1]),
            "ymax": max(bl[1], br[1]),
        }

        canvas = draw_prediction(img_path, pred, gt)
        if canvas is not None:
            cv2.imwrite(str(out / f"{json_path.stem}.png"), canvas)

        # IoU
        ix1 = max(pred["xmin"], gt["xmin"])
        iy1 = max(pred["ymin"], gt["ymin"])
        ix2 = min(pred["xmax"], gt["xmax"])
        iy2 = min(pred["ymax"], gt["ymax"])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        pa = (pred["xmax"]-pred["xmin"]) * (pred["ymax"]-pred["ymin"])
        ta = (gt["xmax"]-gt["xmin"])     * (gt["ymax"]-gt["ymin"])
        iou = max(0, min(1, inter / (pa + ta - inter + 1e-6)))
        ious.append(iou)

        print(f"  {json_path.stem}: IoU={iou:.3f} "
              f"pred_dir={pred['direction']} gt_dir={ann['walking_into_frame_bool']}")

    if ious:
        print(f"\nMean IoU на {len(ious)} train примерах: {sum(ious)/len(ious):.3f}")
        good = sum(1 for x in ious if x > 0.3)
        print(f"IoU > 0.3: {good}/{len(ious)} ({100*good//len(ious)}%)")


if __name__ == "__main__":
    visualize_test(n=10)
    visualize_train(n=20)
    print(f"\nОткрой папку: {VIZ_DIR.absolute()}")
    print("  viz_results/test/  — тестовые (зелёный bbox)")
    print("  viz_results/train/ — train (зелёный=pred, красный=GT)")