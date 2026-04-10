import timm
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

backbone = timm.create_model("efficientnet_b2", pretrained=True, num_classes=0)

model = nn.Sequential(
    backbone,  # 1408-dim features
    nn.Linear(1408, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 6),  # 4 bbox + sin/cos direction
)

img = Image.open("./data/sample_image.png")

output = model(img)

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 16, 128, 128])
    #  torch.Size([1, 24, 64, 64])
    #  torch.Size([1, 48, 32, 32])
    #  torch.Size([1, 120, 16, 16])
    #  torch.Size([1, 352, 8, 8])

    print(o.shape)
