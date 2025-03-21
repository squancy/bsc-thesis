import numpy as np

def ml_est(data):
    c = np.min(data)
    alpha = len(data) / np.sum([np.log(x / c) for x in data])
    return (alpha, c)
