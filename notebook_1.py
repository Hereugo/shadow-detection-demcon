import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Модель
backbone = timm.create_model("efficientnet_b2", pretrained=True, num_classes=0)

model = nn.Sequential(
    backbone,       # → 1408-dim вектор
    nn.Linear(1408, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 6),  # xmin, ymin, xmax, ymax, sin(θ), cos(θ)
)

model.eval()

# Препроцессинг — PIL → тензор
transform = transforms.Compose([
    transforms.Resize((260, 260)),   # EfficientNet-B2 ожидает 260x260
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Загрузка изображения
img = Image.open("./data/sample_image.png").convert("RGB")
tensor = transform(img).unsqueeze(0)  # добавляем batch dimension → [1, 3, 260, 260]

print("Input shape:", tensor.shape)

# Forward pass
with torch.no_grad():
    output = model(tensor)

print("Output shape:", output.shape)   # torch.Size([1, 6])
print("Raw output:", output)
# [xmin, ymin, xmax, ymax, sin(θ), cos(θ)]