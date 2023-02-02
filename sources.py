import numpy as np

def delayed_gaussian_pulse(t0, fmax):
    tau = 0.5 / fmax
    return lambda t: np.exp(-((t - t0) / tau) ** 2)

def delayed_sine(t0, f):
    def func(t):
        y = np.sin(2 * np.pi * f * t)
        if t < t0:
            return np.exp(-((t - t0) ** 2)) * y
        else:
            return y
    return func

"""
Generic TFSF source
"""
class Source:
    def __init__(self, k, f):
        self.k = k
        self.f = f

    def __call__(self, t):
        return self.f(t)

"""
Time delayed gaussian pulse source
"""
class GaussianSource(Source):
    def __init__(self, k, t0, fmax):
        f = delayed_gaussian_pulse(t0, fmax)
        super().__init__(k, f)

"""
Time delayed gaussian mixed sinusoid source
"""
class SineSource(Source):
    def __init__(self, k, t0, f):
        f = delayed_sine(t0, f)
        super().__init__(k, f)

