from typing import *
import numpy as np

def extract_faces(
    img_path: Union[str, np.ndarray],
    gray_scale: bool = False,
    color_face: str = "rgb",
    normalize_face: bool = True,
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None
) -> List[Dict[str, any]]:
    resp = []
    
    img, img_name = image_utils.load_image(img_path)
    
    if img is None:
        raise ValueError(f"Exeption while loading img {img_name}")

    height, width, _ = img.shape
    
    base_region = FacialAreaRegion(x = 0, y = 0, w = width, h = height, confidence = 0)
    
    face_objs = detect_faces(
        img = img,
        max_faces = max_faces
    )
    
    if len(face_objs) == 0:
        if img_name is not None:
            raise ValueError(f"Face could not be detected in {img_name}")
        else:
            raise ValueError(f"Face could not be detected")
    