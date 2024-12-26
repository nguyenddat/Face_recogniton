import os
import bz2
import zipfile
from typing import Optional

import gdown

from folder_utils import get_home
from package_utils import get_tf_major_version

if get_tf_major_version() == 1:
    from keras.models import Sequential
else:
    from tensorflow.keras.models import Sequential

ALLOWED_COMPRESS_TYPES = ["zip", "bz2"]

def download_weights_if_necessary(file_name: str, source_url: str, compress_type: Optional[str] = None) -> None:
    home = get_home()
    target_file = os.path.normpath(os.path.join(home, "/weights", file_name))
    
    if os.path.isfile(target_file):
        return target_file

    try:
        if compress_type is None:
            gdown.download(source_url, target_file, quiet = False)
        elif compress_type is not None and compress_type in ALLOWED_COMPRESS_TYPES:
            gdown.download(source_url, f"{target_file}.{compress_type}", quiet = False)
    except Exception as err:
        raise ValueError(
            "An exception occurred while downloading file"
        ) from err
    
    if compress_type == "zip":
        with zipfile.ZipFile(f"{target_file}.zip", "r") as zip_ref:
            zip_ref.extractall(os.path.join(home, "/weights"))
    elif compress_type == "bz2":
        bz2file = bz2.BZ2File(f"{target_file}.bz2")
        data = bz2file.read()
        with open(target_file, "wb") as f:
            f.write(data)
        
    return target_file

def load_model_weights(model: Sequential, weight_file: str) -> Sequential:
    try:
        model.load_weights(weight_file)
    except Exception as err:
        raise ValueError(
            "An exception occurred while loading the pre-trained weights"
        ) from err
    return model