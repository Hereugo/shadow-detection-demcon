import torch
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from model import ShadowDetector

# ── Конфиг ───────────────────────────────────────────────
TEST_IMG_DIR  = Path("./detection-by-shadow/test_data/test_data")
TEST_JSON_DIR = Path("./detection-by-shadow/test_data/test_data")  # если есть shadow json
ENRICHED_DIR  = Path("./train_data_with_shadow_bbox/train_A_test") # fallback
MODEL_PATH    = "best_model.pth"
OUTPUT_CSV    = "submission.csv"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_shadow_features(img_path: Path, orig_w: int, orig_h: int) -> torch.Tensor:
    """
    Пытаемся найти shadow_bbox для тестового изображения.
    Если нет — возвращаем нули (модель справится, просто менее точно).
    """
    json_path = img_path.with_suffix(".json")

    if json_path.exists():
        with open(json_path) as f:
            ann = json.load(f)

        has_shadow = (
            "shadow_bbox" in ann
            and ann["shadow_bbox"] is not None
            and "xyxy" in ann["shadow_bbox"]
        )

        if has_shadow:
            s = ann["shadow_bbox"]["xyxy"]
            s_xmin = s["xmin"] / orig_w
            s_ymin = s["ymin"] / orig_h
            s_xmax = s["xmax"] / orig_w
            s_ymax = s["ymax"] / orig_h
            s_cx     = (s_xmin + s_xmax) / 2
            s_cy     = (s_ymin + s_ymax) / 2
            s_width  = s_xmax - s_xmin
            s_height = s_ymax - s_ymin

            return torch.tensor([
                s_xmin, s_ymin, s_xmax, s_ymax,
                s_cx, s_cy,
                s_width, s_height,
                0.0, 0.0,  # vec_x, vec_y неизвестны для test
            ], dtype=torch.float32)

    # Нет shadow данных — нули
    return torch.zeros(10, dtype=torch.float32)


def predict():
    print(f"Device: {DEVICE}")

    # Загружаем модель
    model = ShadowDetector()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    print(f"Model loaded: {MODEL_PATH}")

    test_images = sorted(TEST_IMG_DIR.glob("*.png"))
    print(f"Test images: {len(test_images)}")

    results = []

    with torch.no_grad():
        for img_path in test_images:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            # Image tensor
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            # Shadow features
            sf = get_shadow_features(img_path, orig_w, orig_h)
            sf = sf.unsqueeze(0).to(DEVICE)

            # Предсказание
            bbox_pred, dir_pred = model(img_tensor, sf)

            # Денормализуем обратно в пиксели
            xmin = bbox_pred[0, 0].item() * orig_w
            ymin = bbox_pred[0, 1].item() * orig_h
            xmax = bbox_pred[0, 2].item() * orig_w
            ymax = bbox_pred[0, 3].item() * orig_h

            direction = 1 if dir_pred[0, 0].item() > 0.5 else 0

            results.append({
                "id":        img_path.stem,
                "xmin":      round(xmin, 2),
                "ymin":      round(ymin, 2),
                "xmax":      round(xmax, 2),
                "ymax":      round(ymax, 2),
                "direction": direction,
            })

            print(f"\r  {len(results)}/{len(test_images)} — {img_path.name}: "
                  f"bbox=({xmin:.0f},{ymin:.0f},{xmax:.0f},{ymax:.0f}) "
                  f"dir={direction}",
                  end="", flush=True)

    print(f"\n\nГотово! {len(results)} predictions")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    print(f"\nПервые 5 строк:")
    print(df.head().to_string())

    return df


if __name__ == "__main__":
    predict()