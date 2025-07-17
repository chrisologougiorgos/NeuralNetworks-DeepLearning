import numpy as np

class Kernel:
    def __init__(self, degree):
        self.degree = degree


    def calculate_kernel(self, u , v):
        if self.degree == -1:
            gamma = 0.01
            k = np.exp(-gamma * np.linalg.norm(u - v) ** 2)
        elif self.degree == 0:
            k = np.dot(u, v)
        else:
            k = (np.dot(u, v)  + 1)**self.degree
        return k