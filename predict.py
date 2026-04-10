import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms

from model import ShadowDetector


def predict_submission(test_dir: str, model_path: str, output_path: str = "submission.csv"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = ShadowDetector()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)

    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results = []
    test_images = sorted(Path(test_dir).glob("*.png"))

    with torch.no_grad():
        for img_path in test_images:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            tensor = transform(img).unsqueeze(0).to(DEVICE)
            bbox_pred, dir_pred = model(tensor)

            # Денормализуем обратно в пиксели
            xmin = bbox_pred[0, 0].item() * orig_w
            ymin = bbox_pred[0, 1].item() * orig_h
            xmax = bbox_pred[0, 2].item() * orig_w
            ymax = bbox_pred[0, 3].item() * orig_h

            direction = 1 if dir_pred[0, 0].item() > 0.5 else 0

            results.append({
                "id": img_path.stem,
                "xmin": round(xmin, 1),
                "ymin": round(ymin, 1),
                "xmax": round(xmax, 1),
                "ymax": round(ymax, 1),
                "direction": direction,
            })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Submission сохранён: {output_path} ({len(df)} строк)")
    return df


if __name__ == "__main__":
    predict_submission(
        test_dir="./data/test_data",
        model_path="best_model.pth",
        output_path="submission.csv"
    )