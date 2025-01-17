import numpy as np
import tensorflow as tf

def configure_imports():
    np.random.seed(42)
    print("Setting default seed for numpy")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("Enabling memory growth for tensorflow")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
configure_imports()
__all__ = ["np", "tf"]
