import tensorflow as tf

def get_tf_major_version() -> int:
    return int(tf.__version__.split(".", maxsplit = 1)[0])