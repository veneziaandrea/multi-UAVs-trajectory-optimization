import numpy as np

def create_rectangle(x_min, x_max, y_min, y_max):
    rect = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=float)
    return rect

