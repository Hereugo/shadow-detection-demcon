import json
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ShadowDataset(Dataset):
    def __init__(self, json_dir: str, img_dir: str, img_size: int = 260):
        self.img_dir  = Path(img_dir)
        self.img_size = img_size

        self.samples = []
        for json_path in sorted(Path(json_dir).glob("*.json")):
            img_path = self.img_dir / f"{json_path.stem}.png"
            if img_path.exists():
                self.samples.append((img_path, json_path))
            else:
                print(f"  [WARN] No image for {json_path.name}")

        print(f"Найдено {len(self.samples)} samples")
        print(f"  JSON: {json_dir}")
        print(f"  IMG:  {img_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        with open(json_path) as f:
            ann = json.load(f)

        # ── Person bbox (за кадром) ──────────────────────────
        tl = ann["bbox"]["top_left"]
        tr = ann["bbox"]["top_right"]
        bl = ann["bbox"]["bottom_left"]
        br = ann["bbox"]["bottom_right"]

        xs = [tl[0], tr[0], bl[0], br[0]]
        ys = [tl[1], tr[1], bl[1], br[1]]

        xmin = min(xs) / orig_w
        xmax = max(xs) / orig_w
        ymin = min(ys) / orig_h
        ymax = max(ys) / orig_h

        direction = float(ann["walking_into_frame_bool"])

        # ── Shadow bbox ──────────────────────────────────────
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
        else:
            s_xmin = s_ymin = s_xmax = s_ymax = 0.0

        # ── Horizontal flip (50%) ────────────────────────────
        # Самая важная аугментация для нашей задачи:
        # человек слева → человек справа, direction меняется
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Флипаем person bbox относительно центра (1.0)
            xmin, xmax = 1.0 - xmax, 1.0 - xmin

            # Флипаем shadow bbox
            s_xmin, s_xmax = 1.0 - s_xmax, 1.0 - s_xmin

            # Флипаем direction: шёл вправо → теперь влево
            direction = 1.0 - direction

        # ── Применяем transform ──────────────────────────────
        img_tensor = self.transform(img)
        bbox = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

        # ── Shadow features (8 штук, без vec_x/vec_y) ────────
        # vec_x/vec_y убраны потому что на тесте они = 0
        # модель училась на них но не могла использовать → путалась
        if has_shadow:
            s_cx     = (s_xmin + s_xmax) / 2
            s_cy     = (s_ymin + s_ymax) / 2
            s_width  = s_xmax - s_xmin
            s_height = s_ymax - s_ymin

            shadow_features = torch.tensor([
                s_xmin, s_ymin, s_xmax, s_ymax,  # bbox тени   [0:4]
                s_cx, s_cy,                        # центр тени  [4:6]
                s_width, s_height,                 # размеры     [6:8]
            ], dtype=torch.float32)
        else:
            shadow_features = torch.zeros(8, dtype=torch.float32)

        return {
            "image":           img_tensor,
            "bbox":            bbox,
            "direction":       torch.tensor(direction),
            "shadow_features": shadow_features,
            "has_shadow":      torch.tensor(float(has_shadow)),
            "orig_size":       torch.tensor([orig_w, orig_h], dtype=torch.float32),
            "file_name":       ann["file_name"],
        }