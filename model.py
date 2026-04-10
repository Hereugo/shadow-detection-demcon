import timm
import torch
import torch.nn as nn


class ShadowDetector(nn.Module):
    """
    Почему такая архитектура:
    
    1. EfficientNet-B2 backbone (предобученный на ImageNet):
       - Уже умеет видеть формы, края, текстуры
       - Нам нужно только донаучить его видеть тени
       - Выдаёт вектор 1408 признаков
    
    2. Regression head:
       - 4 числа = bbox (xmin, ymin, xmax, ymax) нормализованный
       - БЕЗ sigmoid/relu на выходе — нам нужны числа за пределами [0,1]
         т.к. человек находится ЗА кадром (отрицательные координаты!)
    
    3. Direction head (отдельный):
       - Бинарная классификация: идёт в кадр (1) или из кадра (0)
       - Отдельная голова т.к. это другая задача
    """

    def __init__(self, backbone_name: str = "efficientnet_b2", pretrained: bool = True):
        super().__init__()

        # Backbone — feature extractor
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # убираем оригинальный классификатор
            global_pool="avg"  # global average pooling → вектор
        )

        # Узнаём размер выходного вектора автоматически
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 260, 260)
            feat_dim = self.backbone(dummy).shape[1]
        print(f"Backbone feature dim: {feat_dim}")

        # Bbox regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # xmin, ymin, xmax, ymax — без активации!
        )

        # Direction classification head
        self.dir_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # → вероятность [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)     # [B, 1408]
        bbox = self.bbox_head(features) # [B, 4] — нормализованные координаты
        direction = self.dir_head(features)  # [B, 1] — вероятность
        return bbox, direction