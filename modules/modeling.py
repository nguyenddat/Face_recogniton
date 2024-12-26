from typing import Any

import models.VGGFace as VGGFace

models = {
    "facial_recognition": {
        "VGG-Face": VGGFace.VggFaceClient
    },
    # "spoofing": {
    #     "Fasnet": FasNet.Fasnet,
    # },
    # "facial_attribute": {
    #     "Emotion": Emotion.EmotionClient,
    #     "Age": Age.ApparentAgeClient,
    #     "Gender": Gender.GenderClient,
    #     "Race": Race.RaceClient,
    # },
    # "face_detector": {
    #     "opencv": OpenCv.OpenCvClient,
    #     "mtcnn": MtCnn.MtCnnClient,
    #     "ssd": Ssd.SsdClient,
    #     "dlib": DlibDetector.DlibClient,
    #     "retinaface": RetinaFace.RetinaFaceClient,
    #     "mediapipe": MediaPipe.MediaPipeClient,
    #     "yolov8": YoloFaceDetector.YoloDetectorClientV8n,
    #     "yolov11n": YoloFaceDetector.YoloDetectorClientV11n,
    #     "yolov11s": YoloFaceDetector.YoloDetectorClientV11s,
    #     "yolov11m": YoloFaceDetector.YoloDetectorClientV11m,
    #     "yunet": YuNet.YuNetClient,
    #     "fastmtcnn": FastMtCnn.FastMtCnnClient,
    #     "centerface": CenterFace.CenterFaceClient,
    # }
}

def build_model(task: str, model_name: str) -> Any:
    global cached_models
    if models.get(task) is None:
        raise ValueError(f"Unimplemented task: {task}")
    
    if not cached_models in globals():
        cached_models = {current_task: {} for current_task in models.keys()}
    
    if cached_models[task].get(model_name) is None:
        model = models[task].get(model_name)
        if model:
            cached_models[task][model_name] = model()
        else:
            raise ValueError("Invalid model_name")
    
    return cached_models[task][model_name]