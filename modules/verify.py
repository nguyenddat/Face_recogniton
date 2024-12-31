from typing import Union, List, Optional, Dict, Any, Tuple

import numpy as np

import modeling
import detection


def __extract_faces_and_embeddings(
    img_path: Union[str, np.adarray],
    model_name: str = "VGG-Face",
    anti_spoofing: bool = False
):
    embeddings = []
    facial_areas = []
    
    img_objs = detection.extract_faces(
        img_path = img_path,
        grayscale = False,
        anti_spoofing = anti_spoofing
    )

def verify(
    img1_path: Union[str, np.ndarray, List[float]],
    img2_path: Union[str, np.ndarray, List[float]],
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    threshold: Optional[float] = None,
    anti_spoofing: bool = False
) -> Dict[str, Any]:
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
        img_path: Union[str, np.ndarray, List[float]],
        index: int
    ) -> Tuple[List[List[float]], List[dict]]:
        if isinstance(img_path, list):
            if not all(isinstance(dim, float) for dim in img_path):
                raise ValueError(
                    "Ensure all img_path's items are of type float"
                )
        
            if len(img_path) != dims:
                raise ValueError(
                    f"embeddings should have {dims} dimensions but {index}-th has {len(img_path)} dimensions"
                )
            
            img_embeddings = [img_path]
            img_facial_areas = [no_facial_area]
        else:
            try:
                img_embeddings, img_facial_areas = __extract_faces_and_embeddings(
                    img_path = img_path,
                    model_name = model_name,
                    anti_spoofing = anti_spoofing
                )
            except ValueError as err:
                raise ValueError(f"Exception while processing img{index}_path") from err
        return img_embeddings, img_facial_areas