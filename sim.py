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

        for it in range(self.params.Nt):
            # Update H
            H[:-1] += mH[:-1] * (E[1:] - E[:-1]) / dx
            H[-1] += mH[-1] * (0 - E[-1]) / dx

            # Update E
            E[1:] += mE[1:] * (H[1:] - H[:-1]) / dx
            E[0] += mE[0] * (H[0] - 0) / dx

            # Inject source
            if src := self.source:
                Esrc = src(self.t)
                Hsrc = -src(self.t + dx/(2 * c0) - dt / 2)
                E[src.k] -= mE[src.k] * Hsrc / dx
                H[src.k-1] -= mH[src.k-1] * Esrc / dx

            yield it
            self.t += dt

if __name__ == '__main__':
    params = SimulationParams(Nx=128, dx=0.05, Nt=int(1e3))
    print(params)
    sim = Simulation(params)
    #sim.source = GaussianSource(k=24, t0=(30*params.dx) / c0, fmax=433e6)
    sim.source = SineSource(k=24, t0=(30*params.dx) / c0, f=433e6)
    for it in sim.run():
        print(it, 'min=%e max=%e' % (sim.E.min(), sim.E.max()))
        show_cmap(sim.E, image_size=(640, 480), title='E', wait=int(1e3/60))
        #show_plot(sim.E, sim.H)

