from typing import List, Union
import numpy as np

from FacialRecognition import FacialRecognition
from utils.weight_utils import *

if get_tf_major_version() == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
    )

WEIGHTS_URL = ("https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5")


def l2_normalize(
    x: Union[np.ndarray, list],
    axis: Union[int, None] = None,
    epsilon: float = 1e-10
) -> np.ndarray:
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis = axis, keepdims = True)
    return x / (norm + epsilon)

class VggFaceClient(FacialRecognition):
    def __init__(self):
        self.model = self.load_model()
        self.model_name = "VGG-Face"
        self.input_shape = (224, 224)
        self.output_shape = 4096
    
    def base_model(self) -> Sequential:
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(4096, (7, 7), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(4096, (1, 1), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Convolution2D(2622, (1, 1)))
        model.add(Flatten())
        model.add(Activation("softmax"))

        return model
    
    def load_model(
        self,
        url = WEIGHTS_URL
    ):
        model = self.base_model()
        weight_file = download_weights_if_necessary(
            file_name = "vgg_face_weights.h5",
            source_url = url
        )
        
        model = load_model_weights(
            model = model,
            weight_file = weight_file
        )
        
        base_model_output = Flatten()(model.layers[-5].output)
        vgg_face_descriptor = Model(inputs = model.input, outputs = base_model_output)
        return vgg_face_descriptor
        
    def forward(self, img: np.ndarray) -> List[float]:
        embedding = self.model(img, training = False).numpy()[0].tolist()
        embedding = l2_normalize(embedding)
        return embedding.tolist()