import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ShadowDataset(Dataset):
    """
    Читает пары (image, annotation) из train_data/.
    Каждый sample: image_n.png + image_n.json
    
    JSON содержит:
      - top_left/top_right/bottom_left/bottom_right: пиксели bbox (могут быть отрицательными!)
      - walking_into_frame_bool: 0 или 1
    
    Мы нормализуем bbox на размер изображения чтобы модель
    работала с числами в диапазоне ~[-1, 2] вместо сырых пикселей.
    Это стабилизирует обучение.
    """

    def __init__(self, data_dir: str, img_size: int = 260):
        self.data_dir = Path(data_dir)
        self.img_size = img_size

        # Собираем все PNG файлы у которых есть JSON
        self.samples = []
        for img_path in sorted(self.data_dir.glob("*.png")):
            json_path = img_path.with_suffix(".json")
            if json_path.exists():
                self.samples.append((img_path, json_path))

        print(f"Найдено {len(self.samples)} samples в {data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # ImageNet нормализация — т.к. backbone обучен на ImageNet
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # --- Загружаем изображение ---
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size  # оригинальный размер до ресайза
        img_tensor = self.transform(img)

        # --- Читаем аннотацию ---
        with open(json_path) as f:
            ann = json.load(f)

        # Извлекаем 4 угла bbox в пикселях
        xs = [
            ann["top_left"]["x"], ann["top_right"]["x"],
            ann["bottom_left"]["x"], ann["bottom_right"]["x"]
        ]
        ys = [
            ann["top_left"]["y"], ann["top_right"]["y"],
            ann["bottom_left"]["y"], ann["bottom_right"]["y"]
        ]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Нормализуем на размер изображения
        # Теперь значения ~[-1, 2] вместо [-200, 1000]
        # Модели проще учиться на таких числах
        xmin_n = xmin / orig_w
        xmax_n = xmax / orig_w
        ymin_n = ymin / orig_h
        ymax_n = ymax / orig_h

        direction = float(ann["walking_into_frame_bool"])

        bbox = torch.tensor([xmin_n, ymin_n, xmax_n, ymax_n], dtype=torch.float32)

        return {
            "image": img_tensor,
            "bbox": bbox,                          # [4] нормализованный
            "direction": torch.tensor(direction),  # 0.0 или 1.0
            "orig_size": torch.tensor([orig_w, orig_h], dtype=torch.float32),
            "file_name": ann["file_name"],
        }