from typing import Union

import numpy as np

import modeling as modeling

def verify(
    img1_path,
    img2_path,
    model_name,
    distance_metric,
    threshold,
    anti_spoofing = False
):
    model = modeling.build_model(
        task = "facial_recognition", model_name = model_name
    )
    dims = model.output_shape
    
    no_facial_area = {
        "x": None,
        "y": None,
        "w": None,
        "h": None,
        "left_eye": None,
        "right_eye": None
    }
    
    def extract_embeddings_and_facial_areas(
        img_path,
        index
    ):
        if isinstance(img_path, list):
            pass
