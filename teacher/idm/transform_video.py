import numpy as np
from PIL import Image


def center_crop_to_aspect(image, target_w: int, target_h: int):
    target_ratio = target_w / target_h

    if isinstance(image, Image.Image):
        width, height = image.size
        input_ratio = width / height
        if abs(input_ratio - target_ratio) < 1e-8:
            return image
        if input_ratio < target_ratio:
            crop_w = width
            crop_h = int(round(width / target_ratio))
            left = 0
            top = (height - crop_h) // 2
        else:
            crop_h = height
            crop_w = int(round(height * target_ratio))
            left = (width - crop_w) // 2
            top = 0
        return image.crop((left, top, left + crop_w, top + crop_h))

    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        input_ratio = width / height
        if abs(input_ratio - target_ratio) < 1e-8:
            return image
        if input_ratio < target_ratio:
            crop_w = width
            crop_h = int(round(width / target_ratio))
            left = 0
            top = (height - crop_h) // 2
        else:
            crop_h = height
            crop_w = int(round(height * target_ratio))
            left = (width - crop_w) // 2
            top = 0
        return image[top : top + crop_h, left : left + crop_w]

    raise TypeError(f"Unsupported type for center_crop_to_aspect: {type(image)}")
