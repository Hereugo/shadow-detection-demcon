import numpy as np
import json
from pathlib import Path


# Константы из данных создателя
SUN_ELEVATION_MIN = 10  # градусов
SUN_ELEVATION_MAX = 70
PERSON_ASPECT_RATIO = 2.5  # height / width (стоящий человек)


def estimate_sun_elevation_from_shadow(shadow_length_px: float, shadow_width_px: float) -> float:
    """
    Оцениваем угол солнца из соотношения тени.
    
    person_height ≈ shadow_width * PERSON_ASPECT_RATIO
    tan(elevation) = person_height / shadow_length
    elevation = arctan(person_height / shadow_length)
    
    Клипаем в диапазон 10-70° как сказал создатель данных.
    """
    person_height_est = shadow_width_px * PERSON_ASPECT_RATIO
    
    if shadow_length_px <= 0:
        return 45.0  # fallback
    
    elevation_rad = np.arctan(person_height_est / shadow_length_px)
    elevation_deg = np.degrees(elevation_rad)
    
    # Клипаем по данным создателя
    elevation_deg = np.clip(elevation_deg, SUN_ELEVATION_MIN, SUN_ELEVATION_MAX)
    
    return float(elevation_deg)


def predict_person_bbox(shadow_info: dict, image_w: int, image_h: int) -> dict:
    """
    Главная функция: shadow features → person bbox
    
    Логика:
    1. shadow_width → person_width (одинаковый масштаб перспективы)
    2. person_width × 2.5 → person_height
    3. shadow tip + sun direction → foot position
    4. foot position - person_height → full bbox
    
    shadow_info от Amir:
      xmin, ymin, xmax, ymax  - bbox тени
      tip_x, tip_y            - кончик тени (дальний от человека)
      base_x, base_y          - основание тени (у края кадра ≈ ноги)
      shadow_length_px        - длина тени
      shadow_angle_deg        - угол тени
    """
    
    shadow_w = shadow_info["xmax"] - shadow_info["xmin"]
    shadow_h = shadow_info["ymax"] - shadow_info["ymin"]
    shadow_length = shadow_info.get("shadow_length_px", np.sqrt(shadow_w**2 + shadow_h**2))
    
    # ── ИНСАЙТ: shadow width = person bbox width ──────────────────
    # Перспектива одинаковая т.к. камера фиксированная
    # Чем дальше человек → тень шире в пикселях → человек меньше в пикселях
    # НО: в одной точке пространства масштаб одинаковый
    # Поэтому: person_width_px ≈ shadow_width_px
    person_width_px = shadow_w  # прямое соответствие!
    person_height_px = person_width_px * PERSON_ASPECT_RATIO
    
    # ── Угол солнца ───────────────────────────────────────────────
    sun_elevation = estimate_sun_elevation_from_shadow(shadow_length, shadow_w)
    
    # ── Направление тени → направление к человеку ─────────────────
    # Тень указывает ПРОТИВОПОЛОЖНО от человека
    # base тени (у края кадра) ≈ ноги человека у края
    # tip тени = дальний конец
    
    base_x = shadow_info.get("base_x", shadow_info["xmin"] if shadow_info["tip_x"] > shadow_info["xmin"] else shadow_info["xmax"])
    base_y = shadow_info.get("base_y", shadow_info["ymax"])
    tip_x  = shadow_info.get("tip_x",  shadow_info["xmax"] if shadow_info["tip_x"] > shadow_info["xmin"] else shadow_info["xmin"])
    tip_y  = shadow_info.get("tip_y",  shadow_info["ymin"])
    
    # Вектор: tip → base = направление к человеку
    dx = base_x - tip_x
    dy = base_y - tip_y
    dist = np.sqrt(dx**2 + dy**2) + 1e-6
    direction_x = dx / dist  # нормализованный unit vector
    direction_y = dy / dist
    
    # ── Позиция ног человека ──────────────────────────────────────
    # Ноги = base тени + небольшой offset (тень начинается у ног)
    # shadow_length = person_height / tan(elevation)
    real_shadow_length = person_height_px / np.tan(np.radians(sun_elevation))
    
    # Ноги человека за краем кадра
    foot_x = tip_x + direction_x * real_shadow_length
    foot_y = tip_y + direction_y * real_shadow_length
    
    # ── Строим bbox ───────────────────────────────────────────────
    xmin = foot_x - person_width_px / 2
    xmax = foot_x + person_width_px / 2
    ymax = foot_y          # ноги = низ bbox
    ymin = foot_y - person_height_px  # голова = верх bbox
    
    # ── Определяем direction ──────────────────────────────────────
    # Человек слева (xmax < 0) или справа (xmin > image_w)?
    # Если человек СПРАВА за кадром и движется влево → walking_into_frame = 1
    # Если человек СПРАВА и движется вправо → walking_into_frame = 0
    person_is_right = xmin > image_w / 2  # человек справа от центра
    person_is_left  = xmax < image_w / 2  # человек слева
    
    shadow_moving_right = direction_x > 0  # тень "смотрит" вправо
    
    if person_is_right:
        walking_into_frame = 1 if direction_x < 0 else 0  # идёт влево = в кадр
    elif person_is_left:
        walking_into_frame = 1 if direction_x > 0 else 0  # идёт вправо = в кадр
    else:
        walking_into_frame = -1  # abstain — не уверены
    
    return {
        "xmin": float(xmin),
        "ymin": float(ymin),
        "xmax": float(xmax),
        "ymax": float(ymax),
        "direction": walking_into_frame,
        # debug info
        "_sun_elevation_deg": sun_elevation,
        "_person_width_px": person_width_px,
        "_person_height_px": person_height_px,
        "_foot_x": foot_x,
        "_foot_y": foot_y,
    }


def validate_on_training_sample(json_path: str, shadow_info: dict, image_w: int = 720, image_h: int = 480):
    """
    Проверяем точность на training данных где знаем правильный ответ.
    Считаем IoU между нашим prediction и ground truth.
    """
    with open(json_path) as f:
        ann = json.load(f)
    
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
    
    pred = predict_person_bbox(shadow_info, image_w, image_h)
    
    # IoU
    ix1 = max(pred["xmin"], gt["xmin"])
    iy1 = max(pred["ymin"], gt["ymin"])
    ix2 = min(pred["xmax"], gt["xmax"])
    iy2 = min(pred["ymax"], gt["ymax"])
    
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    pred_area = (pred["xmax"] - pred["xmin"]) * (pred["ymax"] - pred["ymin"])
    gt_area   = (gt["xmax"]  - gt["xmin"])  * (gt["ymax"]  - gt["ymin"])
    union = pred_area + gt_area - inter + 1e-6
    iou = inter / union
    
    print(f"GT bbox:   xmin={gt['xmin']:.0f} ymin={gt['ymin']:.0f} xmax={gt['xmax']:.0f} ymax={gt['ymax']:.0f}")
    print(f"Pred bbox: xmin={pred['xmin']:.0f} ymin={pred['ymin']:.0f} xmax={pred['xmax']:.0f} ymax={pred['ymax']:.0f}")
    print(f"Sun elevation estimated: {pred['_sun_elevation_deg']:.1f}°")
    print(f"IoU: {iou:.3f}")
    
    return iou, pred, gt