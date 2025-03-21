import numpy as np

def mm_est(data):
    X_m = np.mean(data)
    X_m_s = np.mean([x ** 2 for x in data])

    alpha = (np.sqrt(X_m_s ** 2 - X_m_s * X_m ** 2) + X_m_s - X_m ** 2) / (X_m_s - X_m ** 2)
    c = (1 - 1 / alpha) * X_m
    return (alpha, c)
