from PyQt5.QtWidgets import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(320, 400)
        self.main = QWidget()
        self.vbox = QVBoxLayout()

        label = self.push_item(QLabel(self))
        label.setText('EM FDTD Simulation')

        self.inputs = {}

        self.push_textinput('Nx', 'Nx - number of grid cells')
        self.push_textinput('dx', 'dx - spacing of grid cells')
        self.push_textinput('Nt', 'Nt - number of time steps')
        self.push_textinput('freq', 'freq - excitation frequency')
        self.push_textinput('ksrc', 'ksrc - source location')
        self.push_textinput('conf_path', 'Material config file path')

        self.push_checkbox('sine', 'Use sine source instead of gaussian')
        self.push_checkbox('show_cmap', 'Show colormap')
        self.push_checkbox('show_plot', 'Show plot')

        self.btn = self.push_item(QPushButton(self))
        self.btn.setText('Run')
        self.btn.clicked.connect(self.on_run)

        self.main.setLayout(self.vbox)
        self.setCentralWidget(self.main)

    def push_item(self, item):
        self.vbox.addWidget(item)
        return item

    def push_textinput(self, name, placeholder):
        inp = self.push_item(QLineEdit(self))
        inp.setPlaceholderText(placeholder)
        self.inputs[name] = lambda: inp.text()
        return inp

    def push_checkbox(self, name, description):
        inp = self.push_item(QCheckBox(self))
        inp.setText(description)
        self.inputs[name] = lambda: inp.isChecked()
        return inp

    def __getitem__(self, key):
        return self.inputs[key]()

    def on_run(self):
        self.close()

app = QApplication([])
window = MainWindow()
window.show()
app.exec_()

from sim import *
from conf import MaterialConfig

# Parameter definition
Nx = int(window['Nx'])
dx = float(window['dx'])
Nt = int(window['Nt'])

params = SimulationParams(Nx=Nx, dx=dx, Nt=Nt)

# Apply material config if exists
if path := window['conf_path']:
    for (mat, (start, end)) in MaterialConfig(path):
        params.eps[start:end] = mat.eps
        params.mu[start:end] = mat.mu

# Visualization options
should_show_cmap = window['show_cmap']
should_show_plot = window['show_plot']

# Physical length
L = Nx * dx

# Source definition
t0 = (0.4 * L) / c0
ksrc = int(window['ksrc'])
freq = float(window['freq'])
if window['sine']:
    source = SineSource(k=ksrc, t0=t0, f=freq)
else:
    source = GaussianSource(k=ksrc, t0=t0, fmax=freq)

# Setup point history
hist = PointHistory(params.Nt, [0, -1])

# Create simulation
sim = Simulation(params)
sim.source = source

print(params)
input('>')

# Precalc ior for display
ior = (params.eps * params.mu) ** 0.5

for it in sim.run():
    print(it, 'min=%e max=%e' % (sim.E.min(), sim.E.max()))

    # Update point history
    hist.update(sim, it)

    # Visualize fields
    if should_show_cmap:
        show_cmap(sim.E, ior, image_size=(640, 200), title='E', wait=int(1e3/120))
    if should_show_plot:
        show_plot(sim.E, ior, sim.H)

hist.show()

# Show transmission/reflection
s = np.fft.rfftfreq(params.Nt, params.dt)
X = np.fft.rfft(sim.source_hist)
R, T = hist.as_fft(0), hist.as_fft(-1)
plt.plot(s, abs(X), label='source')
plt.plot(s, abs(T), label='transmitted')
plt.plot(s, abs(R), label='reflected')
plt.legend()
plt.show()
TX, RX = (T/X)**2, (R/X)**2
plt.plot(s, TX, label='transmittance')
plt.plot(s, RX, label='reflectance')
plt.legend()
plt.show()

