import numpy as np

def generate_pareto_values(n, alpha, c):
    """Generates n Pareto-distributed values with shape alpha and scale c."""
    u = np.random.uniform(0, 1, n)
    x = c * (1 - u) ** (-1 / alpha)
    return x
