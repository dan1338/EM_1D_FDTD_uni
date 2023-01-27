import numpy as np
from phys import *
from visual import *
from sources import *

class SimulationParams:
    def __init__(self, Nx: int, dx: float, Nt: int, **kwargs):
        self.Nx = Nx
        self.dx = dx
        self.Nt = Nt
        self.dt = kwargs.get('dt', self.find_courant_timestep())
        if not self.satisfies_courant():
            from sys import stderr
            print('@@@ Unstable parameter configuration?', file=stderr)
        self.eps = np.ones(Nx)
        self.mu = np.ones(Nx)

    def find_courant_timestep(self):
        return (self.dx / (2 * c0))

    def satisfies_courant(self):
        return self.dt <= (self.dx / (2 * c0))

    def __repr__(self):
        return f'<SimulationParams Nx={self.Nx} dx={self.dx} Nt={self.Nt} dt={self.dt}>'

class Simulation:
    def __init__(self, params):
        self.params = params
        self.t = 0

        # Fields
        matsize = (params.Nx,)
        self.E = np.zeros(matsize)
        self.H = np.zeros(matsize)

        # TFSF source
        self.source = None

    def run(self):
        # Unpack common vars
        dt, dx = self.params.dt, self.params.dx
        E, H = self.E, self.H

        # Precomputed coefficients
        mH = dt * c0 / self.params.mu
        mE = dt * c0 / self.params.eps

        # Boundary history
        H1, H2, E1, E2 = 0, 0, 0, 0

        for it in range(self.params.Nt):
            # Record boundary
            H2, H1 = H1, H[0]

            # Update H
            H[:-1] += mH[:-1] * (E[1:] - E[:-1]) / dx
            H[-1] += mH[-1] * (E2 - E[-1]) / dx

            # Record boundary
            E2, E1 = E1, E[-1]

            # Update E
            E[1:] += mE[1:] * (H[1:] - H[:-1]) / dx
            E[0] += mE[0] * (H[0] - H2) / dx

            # Inject source
            if src := self.source:
                Esrc = src(self.t)
                Hsrc = -src(self.t + dx/(2 * c0) - dt / 2)
                E[src.k] -= mE[src.k] * Hsrc / dx
                H[src.k-1] -= mH[src.k-1] * Esrc / dx

            yield it
            self.t += dt

class PointHistory:
    def __init__(self, Nt, points):
        self.points = {k: np.zeros(Nt) for k in points}

    def update(self, sim, i):
        for (k, hist) in self.points.items():
            hist[i] = sim.E[k]

    def show(self):
        for (k, hist) in self.points.items():
            plt.plot(hist, label=f'k={k}')
        plt.legend()
        plt.show()

    def as_fft(self, k):
        return np.fft.rfft(self.points[k])

    def __getitem__(self, k):
        return self.points[k]

if __name__ == '__main__':
    params = SimulationParams(Nx=128, dx=0.04, Nt=800)
    params.eps[60:100] = 12.5
    print(params)

    sim = Simulation(params)

    sim.source = GaussianSource(k=24, t0=(30*params.dx) / c0, fmax=433e6)
    #sim.source = SineSource(k=24, t0=(30*params.dx) / c0, f=433e6)

    hist = PointHistory(params.Nt, [0, -1])

    input('>')

    for it in sim.run():
        print(it, 'min=%e max=%e' % (sim.E.min(), sim.E.max()))

        # Update point history
        hist.update(sim, it)

        # Visualize fields
        show_cmap(sim.E, image_size=(640, 200), title='E', wait=int(1e3/120))
        #show_plot(sim.E, sim.H)

    hist.show()

