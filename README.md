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
