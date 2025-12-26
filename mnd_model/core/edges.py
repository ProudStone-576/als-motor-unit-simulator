import numpy as np

class Edge:
    def __init__(self, src, dst, attenuation=1.0, noise=0.0):
        self.src = src
        self.dst = dst
        self.attenuation = attenuation
        self.noise = noise

    def transmit(self):
        noise_term = np.random.normal(0, self.noise)
        return self.src.output_signal * self.attenuation + noise_term
