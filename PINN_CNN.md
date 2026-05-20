# Physics-Informed Neural Network (PINN) for Drug Concentration

A simple PyTorch implementation of a Physics-Informed Neural Network (PINN) to solve a first-order pharmacokinetic ODE. This project demonstrates how to enforce physical laws (differential equations) directly into the loss function of a neural network.

<p align="center">
  <img src="workflow.png" width="100%">
</p>

## Problem Setup

We model the concentration of a drug in the bloodstream, $C(t)$, which decays over time following a simple first-order differential equation:

$$
\frac{dC}{dt} = -kC
$$

Subject to the initial condition:

$$
C(0) = C_0
$$ 

Where:
- $C(t)$ is the drug concentration at time $t$.
- $k$ is the elimination rate constant.
- $C_0$ is the initial concentration.

Instead of training purely on data points, the neural network $NN(t; \theta)$ minimizes a composite loss function:
1. **Boundary Loss**: Ensures $NN(0) \approx C_0$.
2. **Physics Loss**: Enforces the residual $\left( \frac{dNN}{dt} + k \cdot NN \right)^2 \approx 0$ across the time domain using automatic differentiation.

## Usage

Run the standalone script to train the model and generate the loss plot and concentration animation:

```bash
python drug_pinn.py
```

This will produce:
- `drug_pinn_concentration.gif`: An animation of the learned solution.
- `drug_pinn_loss.png`: The training loss history.

## References

For more reading, see:
> Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations**. *Journal of Computational Physics*, 378, 686–707.

---
*Created by Zara Darcy and assisted by Cursor AI.*
