from abc import ABC
from typing import Any, Union, List, Tuple
import numpy as np

from utils.package_utils import get_tf_major_version

if get_tf_major_version() == 2:
    from tensorflow.keras.models import Model
else:
    from keras.models import Model

class FacialRecognition(ABC):
    model: Union[Model, Any]
    model_name: str
    input_shape: Tuple[int, int]
    output_shape: int
    
    def forward(self, img: np.ndarray) -> List[float]:
        if not isinstance(self.model, Model):
            raise ValueError(
                "Must overwrite forward method if it is not a keras model"
            )
        
        return self.model(img, training = False).numpy()[0].tolist()
