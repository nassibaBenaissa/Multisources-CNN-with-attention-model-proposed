# Multisources-CNN-with-attention-model-proposed
how to use attention with a CNN physics learned model using a multisources datasets, here are used few data layers ( SST, SSH, POC ....) , can ad more layers.
#Cell 1: Environment & Data Input
# Cell 1: Environment & Imports
!pip install torch numpy matplotlib tqdm -q

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import torch.autograd as autograd

# Default configuration for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 👇 DATA INPUT LINE: Load external observations (CSV, NetCDF, or NumPy)
# Example: Load depth-profile observations for validation or data assimilation
# obs_data = np.loadtxt('npz_observations.csv', delimiter=',', skiprows=1)
# N_obs, P_obs, Z_obs = torch.tensor(obs_data[:, 0]), torch.tensor(obs_data[:, 1]), torch.tensor(obs_data[:, 2])


#Cell 2: Model Definition
# Cell 2: Model Definition
class DifferentiableNPZ(nn.Module):
    """
    Fully differentiable 1D vertical NPZ model.
    
    Implemented processes:
    • Nutrient limitation: μ(N) = μ_max·N/(k_N+N) [Michaelis & Menten, 1913; Dugdale, 1967]
    • Zooplankton grazing: g·P·Z [Fasham et al., 1990]
    • Turbulent diffusion: Kz·∂²C/∂z² [Evans & Parslow, 1985; Franks et al., 1986]
    
    All parameters are nn.Parameter → native gradients via torch.autograd.
    """
    def __init__(self, dz=1.0, mu_max=0.8, k_N=0.5, g=0.6, gamma=0.7, m_P=0.1, m_Z=0.15, Kz=1e-4):
        super().__init__()
        self.dz = dz
        self.mu_max = nn.Parameter(torch.tensor(mu_max))
        self.k_N    = nn.Parameter(torch.tensor(k_N))
        self.g      = nn.Parameter(torch.tensor(g))
        self.gamma  = nn.Parameter(torch.tensor(gamma))
        self.m_P    = nn.Parameter(torch.tensor(m_P))
        self.m_Z    = nn.Parameter(torch.tensor(m_Z))
        self.Kz     = nn.Parameter(torch.tensor(Kz))

    def laplacian(self, C: torch.Tensor) -> torch.Tensor:
        """Vertical diffusion Kz·∂²C/∂z² (2nd-order centered scheme + Neumann BCs)"""
        d2C = torch.zeros_like(C)
        d2C[1:-1] = (C[2:] - 2*C[1:-1] + C[:-2]) / self.dz**2
        d2C[0]    = (C[1] - C[0]) / self.dz**2  # ∂C/∂z ≈ 0 at surface
        d2C[-1]   = (C[-2] - C[-1]) / self.dz**2 # ∂C/∂z ≈ 0 at bottom
        return d2C

    def forward(self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        N, P, Z = state
        
        # 1. Michaelis-Menten (nutrient limitation)
        mu = self.mu_max * (N / (self.k_N + N + 1e-9))
        
        # 2. gZ grazing (linear in P and Z)
        grazing = self.g * P * Z
        assimilation = self.gamma * grazing
        excretion    = (1.0 - self.gamma) * grazing
        
        # 3. Turbulent diffusion Kz
        diff_N = self.Kz * self.laplacian(N)
        diff_P = self.Kz * self.laplacian(P)
        diff_Z = self.Kz * self.laplacian(Z)
        
        # Differentiable mass balances
        dNdt = -mu*P + excretion + self.m_P*P + self.m_Z*Z + diff_N
        dPdt =  mu*P - grazing - self.m_P*P + diff_P
        dZdt =  assimilation - self.m_Z*Z + diff_Z
        
        return dNdt, dPdt, dZdt

        #Cell 3: Native PyTorch RK4 Solver
        # Cell 3: Native PyTorch RK4 Solver
def rk4_step(model: nn.Module, state: tuple, dt: float) -> tuple:
    """4th-order Runge-Kutta fully traceable by autograd"""
    k1 = model(state)
    s2 = [s + dt/2 * k for s, k in zip(state, k1)]
    k2 = model(s2)
    s3 = [s + dt/2 * k for s, k in zip(state, k2)]
    k3 = model(s3)
    s4 = [s + dt * k for s, k in zip(state, k3)]
    k4 = model(s4)
    return [s + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i, s in enumerate(state)]

    #Cell 4: Simulation Execution
    # Cell 4: Simulation Execution
nz = 50                     # vertical layers
depth = np.linspace(0, 50, nz) # profile 0-50m
dz = depth[1] - depth[0]
dt = 1800.0                 # time step: 30 min
n_days = 7
steps_per_day = 48
total_steps = n_days * steps_per_day

model = DifferentiableNPZ(dz=dz, mu_max=0.6, k_N=0.8, g=0.5, Kz=2e-4)

# Initial conditions
N0 = torch.full((nz,), 10.0)
P0 = torch.full((nz,), 0.1)
P0[20:30] = 2.0             # initial subsurface bloom
Z0 = torch.full((nz,), 0.2)

state = (N0, P0, Z0)
history = {"N": [N0.detach().numpy()], "P": [P0.detach().numpy()], "Z": [Z0.detach().numpy()]}

for _ in tqdm(range(total_steps), desc="🌊 Biogeochemical simulation"):
    state = rk4_step(model, state, dt)
    history["N"].append(state[0].detach().numpy())
    history["P"].append(state[1].detach().numpy())
    history["Z"].append(state[2].detach().numpy())

    #Cell 5: Visualization
    # Cell 5: Plotting
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, key, title in zip(axes, ["N", "P", "Z"], ["Nutrients (N)", "Phytoplankton (P)", "Zooplankton (Z)"]):
    data = np.array(history[key]).T
    im = ax.imshow(data, aspect='auto', extent=[0, n_days, depth[0], depth[-1]], 
                   origin='upper', cmap='viridis' if key != "N" else 'magma')
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel('Time (days)')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label='Concentration (μmol N·m⁻³)')

plt.suptitle("Spatio-temporal NPZ 1D Evolution (Differentiable PyTorch)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#Cell 6: Differentiability & Gradient Test
# Cell 6: Proof of concept: gradient-based optimization
print("🔍 Testing differentiability...")

# New model for demo (avoids computational graph sharing)
model_opt = DifferentiableNPZ(dz=dz, mu_max=0.6, k_N=0.8, g=0.5, Kz=2e-4)
state_opt = (N0.clone(), P0.clone(), Z0.clone())

# Forward pass for 24 steps
for _ in range(24):
    state_opt = rk4_step(model_opt, state_opt, dt)

# Loss function: maximize integrated phytoplankton biomass
loss = -state_opt[1].sum()  # negative sign because optimizers minimize loss
loss.backward()

print("✅ Gradient successfully computed via torch.autograd!")
print(f"   ∂Loss/∂μ_max = {model_opt.mu_max.grad:.5f}")
print(f"   ∂Loss/∂Kz    = {model_opt.Kz.grad:.6e}")
print(f"   ∂Loss/∂g     = {model_opt.g.grad:.5f}")

#Cell 7: References & Extensions
# 📖 Integrated Bibliographical References
| Component | Reference | Role in code |
|-----------|-----------|--------------|
| **Michaelis-Menten** | Michaelis & Menten (1913) *Biochem. Z.*; Dugdale (1967) *J. Mar. Res.* | `mu = μ_max·N/(k_N+N)` |
| **Grazing `gZ`** | Fasham, Ducklow & McKelvie (1990) *J. Mar. Res.* | `grazing = g·P·Z` |
| **Diffusion `Kz`** | Evans & Parslow (1985) *Deep-Sea Res.*; Franks et al. (1986) | `Kz·laplacian(C)` |
| **Differentiability** | Chen et al. (2018) *NeurIPS* (Neural ODE); Raissi et al. (2019) | Native `torch.autograd` |

## 🔧 Recommended Extensions
1. **Light coupling**: Add `I(z,t) = I₀·exp(-k_w·z - k_P·∫P dz)` → `μ = μ_max·I/(k_I+I)·N/(k_N+N)`
2. **Non-linear functional response**: `g·P/(k_P+P)·Z` (Holling Type II) for grazing saturation
3. **Data assimilation**: `loss = torch.sum((P_sim - P_obs)**2)` + `optimizer = torch.optim.LBFGS`
4. **Large-scale production**: Replace `rk4_step` with `torchdiffeq.odeint_adjoint` for efficient adjoint solver and adaptive time-stepping.
