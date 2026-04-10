# Winning BrabantHack_26: shadow-based pedestrian detection
Your fastest path to victory combines a pre-trained CNN backbone with a custom regression
head that directly predicts off-screen bounding boxes from shadow features, wrapped in a
Gradio dashboard that makes the invisible visible. This approach can yield a working
prototype within hours rather than days, leaving time for polish, metrics, and a pitch that frames
shadow detection as a privacy-preserving safety technology — a narrative tailor-made for
Demcon's synthetic data expertise and European values. What follows is a complete blueprint
covering architecture selection, pipeline design, training strategy, demo construction, and pitch
structure for what is a one-day hackathon at High Tech Campus Eindhoven on April 10, 2026.

The core technical problem and why geometry is your secret weapon
This challenge sits at the intersection of shadow analysis, geometric reasoning, and bounding
box regression. The fundamental insight: a shadow encodes the 3D position of the person
casting it through well-understood projective geometry. Given a shadow visible in-frame and
knowledge (or estimation) of the light source, a person's foot position can be computed as:

Shadow Length = Person Height / tan(Sun Elevation)

Shadow Direction = Sun Azimuth + 180°

Person Foot Position = Shadow Tip − (Shadow Length × Direction Vector)

This means the problem is not purely a black-box regression task — there is strong geometric
structure you can exploit. The winning approach combines learned features (a CNN that
implicitly captures shadow geometry from data) with explicit geometric priors (encoding
shadow orientation, length, and vanishing point relationships).
The most directly relevant academic work is Liu et al., CVPR 2023, "What You Can Reconstruct
from a Shadow" (Columbia University), which explicitly reconstructs occluded 3D objects from
their cast shadows using a differentiable image formation model. The SOBA dataset (CVPR
2020) provides 3,623 shadow-object pairs with instance-level annotations linking each shadow
to its corresponding object — exactly the type of association your model must
learn. The SSIS/SSISv2 framework (CVPR 2021 Oral, TPAMI 2022) already predicts offset
vectors from shadow centroids to object centroids, which is mechanically
identical to your task. These papers and datasets provide both theoretical grounding and
practical starting points.

The recommended architecture: two paths ranked by speed

## Path A — Fastest to implement (recommended): CNN backbone + regression head

This is your 4-hour MVP. Use a pre-trained EfficientNet-B2 or ResNet-50 from the timm library
REWIN
arXiv TheCVF
arXiv Papers with Code
as a feature extractor, then attach a lightweight multi-task head that outputs 6 values: 4
bounding box coordinates (center_x, center_y, width, height) and 2 direction values (sin θ, cos θ
for walking angle).

The critical design decision is the coordinate system for off-screen predictions. Since the
person is outside the camera frame, use an extended normalized coordinate space: the image
occupies [0, 1] × [0, 1], and predictions can range across [−1, 2] × [−1, 2], allowing localization up to one full image-width beyond each edge. Avoid sigmoid activations on the output — use unconstrained regression to allow off-screen values. Train with CIoU loss for the bounding box (it penalizes center distance, overlap, and aspect ratio simultaneously, outperforming
L1/L2 by 0.5–1.5% AP on standard benchmarks) and MSE loss on sin/cos for direction. Combine
as: L = λ₁ × L_bbox + λ₂ × L_direction , starting with λ₁ = 1.0 and λ₂ = 0.5.

## Path B — More powerful if time permits: YOLO-Pose adaptation

YOLO-Pose (YOLOv8-pose or YOLO11-pose from Ultralytics) already has a dual detection head
that predicts both bounding boxes and keypoint coordinates via regression. Treat the shadow as
the detected object and the off-screen person location as "keypoints" (4 corner coordinates). This
gives you object detection on the shadow plus regression to the person location in a single
forward pass at >100 FPS for the nano variant. The Ultralytics ecosystem provides built-in
ONNX/TensorRT export, making real-time deployment trivial.

Why not transformers or DETR?

Vision transformers (ViT, Swin, DETR) offer excellent global reasoning — valuable for
understanding long-range spatial relationships between shadow features and off-screen
positions. However, they converge slowly and demand more training data. For a one-day
hackathon, the CNN regression approach dominates on implementation speed while delivering
competitive accuracy. If you have extra time late in the day, swapping in a ConvNeXt-Tiny
backbone (transformer-like performance with CNN training stability) is a low-risk upgrade.
Building the ML pipeline: end-to-end beats two-stage
The temptation is to build a multi-stage pipeline: shadow segmentation → geometric feature

```python
import timm, torch.nn as nn
backbone = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
model = nn.Sequential(
    backbone, # 1408-dim features
    nn.Linear(1408, 512), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(512, 6) # 4 bbox + sin/cos direction
)
```
arXiv
extraction → bounding box regression. Resist this for your MVP. An end-to-end model that
takes the raw image and directly predicts the bounding box is faster to implement, easier to
debug, and avoids error propagation between stages. The CNN will implicitly learn to detect
shadow regions and extract geometric features during training.
However, if the provided synthetic dataset includes shadow masks as part of the annotations,
exploit them as auxiliary input. Concatenate the shadow mask as a 4th channel (RGB + mask →
4-channel input), which gives the model an explicit signal about where the shadow is. Modify
the first convolutional layer to accept 4 channels instead of 3 by copying the pre-trained weights
and initializing the 4th channel weights to zero.
For walking direction prediction, the sin/cos parameterization is essential — it avoids the
discontinuity at the 0°/360° boundary that breaks standard angle regression. At inference,
recover the angle via θ = atan2(sin_pred, cos_pred) . An alternative is coarse classification
into 8 compass directions (N, NE, E, SE, S, SW, W, NW), which is simpler to train and may be
sufficient for the judges. Shadow orientation provides the strongest signal: fit a line through
shadow pixels using cv2.fitLine() or compute PCA on shadow coordinates — the principal
axis aligns with walking direction.

Taming the synthetic-to-real domain gap

Since the challenge uses synthetic data, bridging the domain gap is make-or-break. Three
techniques ranked by hackathon-friendliness:

Domain randomization during training is the single most impactful technique. NVIDIA's
research shows that DR alone achieves 75–99% mAP@50 on synthetic-to-real transfer
when done well. The key parameters to randomize are lighting direction/intensity/color,
background textures, shadow intensity/softness, camera angle, and post-processing
(brightness ±40%, contrast ±30%, Gaussian noise, JPEG artifacts). Research consistently
finds that variety of subjects/objects matters most, followed by lighting variation.

Aggressive data augmentation during training serves as implicit domain adaptation.
Apply random horizontal flips (mirror the direction label too), rotation ±15°, color jitter,
CutMix (α=1.0, which outperforms MixUp for localization), and mosaic
augmentation (YOLO-style, combining 4 images). Shadow-specific augmentations include
varying shadow edge sharpness (Gaussian blur on shadow boundaries) and random
shadow intensity adjustment to simulate overcast vs. harsh sunlight.

Style transfer as a stretch goal: if time permits, use Contrastive Unpaired Translation
(CUT) to translate synthetic shadow images toward a real-world appearance. CUT is faster
and more stable than CycleGAN. Even 10–20 real shadow images as target domain
reference can close a substantial portion of the gap when combined with self-training
(predict on unlabeled real images, add high-confidence predictions as pseudo-labels,
retrain).
Medium
arXiv

Training strategy matters as much as augmentation. Use transfer learning from
ImageNet/COCO pre-trained weights — this reduces required training data by 5–10× versus
training from scratch. Freeze the backbone for 5–10 epochs while training only the regression
head, then unfreeze with a 10× lower learning rate and layer-wise decay. Use AdamW
optimizer with cosine annealing schedule, starting at lr=1e-4 for the head and 1e-5 for the
backbone.

Avoiding the eight shadow detection pitfalls that sink teams
Shadow-based CV has well-documented failure modes that will differentiate informed teams
from the rest:
1. Dark objects misclassified as shadows — the #1 failure. Shadows preserve the underlying
ground texture while dark objects do not. Working in YCbCr or Lab* color space (which
separates luminance from chrominance) dramatically reduces false positives.
2. Soft/diffuse shadows under overcast skies vanish for models trained only on harsh
shadows. Augment with variable shadow edge blur.
3. Multiple overlapping shadows from multiple light sources or multiple people create
ambiguous inputs. The model must handle multi-shadow scenes gracefully.
4. Perspective distortion causes shadow shapes to elongate dramatically at low sun angles or
oblique camera views.
5. Shadow merging where adjacent shadows fuse into one connected region, confounding
single-person assumptions.
6. Ground texture variation — shadows on grass look fundamentally different from shadows
on concrete. Ensure training data covers diverse surfaces.
7. Scale sensitivity — small or distant shadows disappear at low resolution. Use multi-scale
feature extraction (FPN-style).
8. Annotation noise — even benchmark datasets like SBU contain significant labeling errors.
The synthetic dataset should have clean labels, but validate a sample before
training.
Acknowledging these limitations in your pitch — and showing you've mitigated some — signals
deep domain expertise to judges.
The demo that wins: a four-panel Gradio dashboard
Use Gradio, not Streamlit. Gradio has built-in bounding box annotation components, WebRTC
support for near-zero-latency video processing, and deploys to HuggingFace Spaces with
one click. A basic CV demo takes under 20 lines of code, freeing time for the model itself.
LinkedIn
ResearchGate
ResearchGate
ScienceDirect
Gradio
Design a four-panel layout that makes the prediction visually intuitive:
1. Input panel (top-left): The shadow image with detected shadow region highlighted via
semi-transparent purple overlay. This shows the model "sees" the shadow correctly.
2. Extended canvas prediction (top-right): The image expanded by 50–100% on each side,
with the predicted bounding box drawn in green at the off-screen location and a
directional arrow showing predicted walking direction. This is the money shot — making
the invisible person visible.
3. Metrics panel (bottom-left): Real-time IoU score, FPS counter, confidence gauge, and a
running accuracy chart across test samples.
4. Ground truth comparison (bottom-right): Toggle overlay showing predicted bbox (red) vs.
ground truth (green) vs. overlap (yellow), with a slider to scrub through test examples.
Add a bird's-eye-view mini-map showing the camera's field of view as a cone, shadow positions
as dark shapes, and predicted pedestrian positions as colored dots with direction arrows. This
top-down schematic makes the spatial reasoning immediately legible to non-technical judges.
For walking direction, draw thick directional arrows emanating from the predicted person
center, with arrow length proportional to confidence. Use a color gradient (green = high
confidence, red = low confidence). If temporal data is available, show a predicted trajectory as
fading dots.
The pitch structure that impresses Demcon
BrabantHack is a single-day event — you pitch at the end. Demcon is a Dutch high-tech
engineering company that specializes in synthetic data for machine vision. Your
roadmap should directly leverage their capabilities. Here is the optimal 4-minute pitch:
Problem hook (30 seconds): "Every year 270,000 pedestrians are killed on roads globally.
Current ADAS systems detect people only when they're already in the camera frame. But what if
we could see them coming — by reading their shadows?"
Live demo (90 seconds): Walk through 3–4 test images in the Gradio dashboard. Show the
shadow, the predicted off-screen location, the walking direction arrow, and the IoU scores. Let a
judge drag in a test image if possible. Have a pre-recorded backup video — live demos at
hackathons fail frequently.
Technical approach (45 seconds): "We use an EfficientNet backbone fine-tuned on the
synthetic dataset, with a multi-task head predicting both bounding box coordinates and walking
direction. CIoU loss for localization, sin/cos regression for direction. The model runs at X FPS —
real-time capable."
Product roadmap (45 seconds): Present a visual timeline with four phases:
Demcon
Phase 1 (0–3 months): MVP with synthetic data, single-camera controlled lighting, IoU >
0.3
Phase 2 (3–9 months): Multi-shadow handling, diverse lighting, real-world data collection,
edge deployment on Jetson Orin, IoU > 0.5
Phase 3 (9–18 months): ADAS integration via CAN bus, multi-sensor fusion with
LiDAR/radar, ISO 26262 compliance, surveillance SDK
Phase 4 (18–36 months): Smart city platform, retail analytics, defense/perimeter security
Demcon alignment (30 seconds): "This builds directly on Demcon's synthetic data pipeline —
generating procedural 3D environments with controlled lighting to produce unlimited training
data. And shadow-based detection is inherently privacy-preserving — no faces, no
biometrics, fully GDPR-compliant. It's a European approach to safetyAI."
The privacy angle is your strategic differentiator. Shadow detection captures zero biometric data,
aligning perfectly with the EU AI Act and GDPR. This resonates deeply with European judges
and with Demcon's defense/security market verticals.
Hour-by-hour execution plan for a one-day hackathon
Assuming a roughly 8-hour working day with final pitches:
Hours 0–2 (Foundation): Set up the environment (PyTorch, timm, Ultralytics, Gradio, OpenCV,
supervision). Explore the provided dataset — understand image format, annotation schema,
shadow characteristics, coordinate system. Get a baseline model running: load pre-trained
EfficientNet-B2, attach regression head, run a single forward pass on one image. Assign team
roles: ML engineer (model), CV/data engineer (data pipeline + augmentation), demo developer
(Gradio dashboard), presenter (slides + roadmap from hour 1).
Hours 2–5 (Core development): Train the model. Start with 20–30 epochs, evaluate on
validation split. Implement CIoU loss, augmentation pipeline, and direction prediction head.
Simultaneously, the demo developer builds the Gradio 4-panel layout with placeholder
visualizations. The presenter drafts the pitch and product roadmap slides.
Hours 5–7 (Integration and improvement): Connect the trained model to the Gradio dashboard.
Run evaluation metrics (IoU, direction accuracy, FPS). If accuracy is low, try: adding shadow
mask as 4th input channel, switching to YOLO-Pose approach, adding geometric features
(shadow orientation from OpenCV). Record a backup demo video.
Hours 7–8 (Polish and pitch): Finalize the dashboard, prepare 3–4 compelling test examples,
rehearse the pitch twice, prepare for Q&A on limitations (overcast weather, merged shadows,
small zenith angles, synthetic-to-real gap).
Demcon
The complete recommended tech stack
Component Tool Why
Deep learning
framework
PyTorch Fastest prototyping, largest
ecosystem
Vision backbone timm (EfficientNet-B2 or ConvNeXtTiny)
900+ pre-trained models, one-line
loading
Alternative approach Ultralytics YOLO11-pose Built-in keypoint regression, realtime
Shadow segmentation SAM2 + point prompts or OpenCV
thresholding
Quick shadow isolation if needed
Bbox loss function CIoU Best balance of convergence and
accuracy
Geometric analysis OpenCV (fitLine, PCA, minAreaRect) Shadow orientation extraction
Visualization Roboflow Supervision Beautiful bbox overlays in 2 lines
Demo/dashboard Gradio Blocks WebRTC video, native bbox
support
Export/optimization ONNX → TensorRT (via Ultralytics or
torch.onnx)
Real-time inference proof
Augmentation Albumentations + Mosaic + CutMix Comprehensive, fast, shadowfriendly
Conclusion: what separates winners from participants
The teams that lose will spend 6 hours building a complex multi-stage pipeline that barely works
by pitch time. The teams that win will have a simple, working model within 3 hours and spend
the remaining time making the demo beautiful, the metrics compelling, and the pitch razorsharp. Start with the simplest approach that could possibly work (CNN + regression head),
validate it produces reasonable outputs, then iterate. The EfficientNet backbone already
understands visual features from ImageNet — it needs surprisingly little fine-tuning to learn
shadow geometry from synthetic data.
Three non-obvious edges to exploit: first, the SSIS codebase (GitHub: stevewongv/SSIS) already
predicts shadow-to-object offset vectors and is built on Detectron2 — if the dataset format
is compatible, this gives you a massive head start. Second, explicit geometric features (shadow
arXiv
orientation angle and length extracted via OpenCV PCA) concatenated with CNN features
create a powerful hybrid that outperforms either alone. Third, the privacy-preserving narrative
transforms a technical demo into a product story that resonates with European judges evaluating
real-world impact. Demcon wants to see systems-engineering thinking and a credible path from
prototype to production Demcon Demcon — give them exactly that.