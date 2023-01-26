from dataclasses import dataclass

# Speed of light
c0 = 299792458 # m/s

# Freespace permittivity
eps0 = 8.8541e-12 # F/m

# Freespace permeability
mu0 = 1.2566e-6 # H/m

# Freespace impedance
Z0 = (mu0 / eps0) ** 0.5

@dataclass
class Material:
    # Relative permittivity
    eps: float
    # Relative permeability
    mu: float
    # Wave impedance
    @property
    def Z(self):
        return ((mu0 * self.mu) / (eps0 * self.eps)) ** 0.5

# Dielectric mediums
materials = {
    'Air': Material(eps=1.000589, mu=1),
    'PTFE': Material(eps=2.1, mu=1),
    'Polyethylene' : Material(eps=2.25, mu=1),
    'Polystyrene': Material(eps=2.5, mu=1),
    'Polyimide': Material(eps=3.4, mu=1),
    'Quartz': Material(eps=3.9, mu=1),
    'Sapphire': Material(eps=8.9, mu=0.999),
    'Graphite': Material(eps=12.5, mu=0.999),
    'Water': Material(eps=80.2, mu=0.999)
}

