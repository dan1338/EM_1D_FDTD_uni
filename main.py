from sim import *

# Parameter definition
Nx = 128
dx = 0.05
Nt = 1000

params = SimulationParams(Nx=Nx, dx=dx, Nt=Nt)

# Physical length
L = Nx * dx

sim = Simulation(params)

# Source definition
t0 = (0.2 * L) / c0
freq = 433e6
#sim.source = GaussianSource(k=24, t0=t0, fmax=freq)
sim.source = SineSource(k=24, t0=t0, f=freq)

# Material parameters
params.set_material(slice(60, 100), 'PTFE')
#params.set_material(slice(10, 20), 'Graphite')

# Setup point history
hist = PointHistory(params.Nt, [0, -1])

print(params)
input('>')

for it in sim.run():
    print(it, 'min=%e max=%e' % (sim.E.min(), sim.E.max()))

    # Update point history
    hist.update(sim, it)

    # Visualize fields
    show_cmap(sim.E, image_size=(640, 200), title='E', wait=int(1e3/120))
    #show_plot(sim.E, sim.H)

hist.show()

# Show transmission/reflection
s = np.fft.rfftfreq(params.Nt, params.dt)
R, T = hist.as_fft(0), hist.as_fft(-1)
plt.plot(s, abs(T), label='transmitted')
plt.plot(s, abs(R), label='reflected')
plt.legend()
plt.show()

