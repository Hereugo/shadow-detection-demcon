import timm
import torch
import torch.nn as nn


class ShadowDetector(nn.Module):
    """
    Два потока входных данных:
    
    1. Image stream: EfficientNet-B2 → 1408 features
       Учится видеть тень в изображении
    
    2. Shadow stream: shadow_bbox features → 64 features  
       Явные геометрические данные от Amir:
       [xmin, ymin, xmax, ymax, cx, cy, width, height, vec_x, vec_y]
    
    Объединяем оба потока → предсказываем person bbox
    
    Почему два потока лучше одного:
    - CNN видит текстуры и формы, но медленно учит геометрию
    - Shadow features = прямая геометрия, быстро обучается
    - Вместе: CNN исправляет ошибки геометрики и наоборот
    """

    def __init__(self, backbone_name: str = "convnext_tiny",
                 pretrained: bool = True,
                 shadow_feat_dim: int = 8):
        super().__init__()

        # ── Image backbone ───────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 260, 260)
            feat_dim = self.backbone(dummy).shape[1]

        print(f"Backbone feature dim: {feat_dim}")

        # ── Shadow feature encoder ───────────────────────────
        # 10 shadow features → 64 dim representation
        self.shadow_encoder = nn.Sequential(
            nn.Linear(shadow_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # ── Fusion: image + shadow → combined ───────────────
        combined_dim = feat_dim + 64

        # ── BBox head ────────────────────────────────────────
        # Нет финальной активации! Координаты могут быть отрицательными
        self.bbox_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),   # xmin, ymin, xmax, ymax
        )

        # ── Direction head ───────────────────────────────────
        self.dir_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),        # → вероятность [0,1]
        )

    def forward(self, image, shadow_features):
        # Image features
        img_feat    = self.backbone(image)          # [B, 1408]

        # Shadow features
        shadow_feat = self.shadow_encoder(shadow_features)  # [B, 64]

        # Объединяем
        combined = torch.cat([img_feat, shadow_feat], dim=1)  # [B, 1472]

        bbox      = self.bbox_head(combined)   # [B, 4]
        direction = self.dir_head(combined)    # [B, 1]

        return bbox, direction