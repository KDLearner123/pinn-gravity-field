# 🌌 PINN Space Engine: Gravitational Field Meta-Solver (V1)

A Physics-Informed Neural Network (PINN) designed to solve Laplace's Equation for gravitational potential fields ($\nabla^2 U = 0$) using **JAX**, **Equinox**, and **Optax**. 

Instead of treating field estimation as a pure data-regression task, this architecture blends analytical boundary conditions with automatic differentiation (AD) constraints to learn a continuous, mesh-free mapping of space.

---

## 🚀 The Core Challenge & Architecture Evolution

Modeling a Newtonian point mass singularity ($U = -1/r$) within a standard Cartesian network introduces massive numerical hurdles. This repository showcases the engineering steps taken to overcome them:

### 1. The "Ghost Sheet" Problem (Spectral Bias)
Neural networks suffer from **spectral bias**, inherently prioritizing low-frequency, smooth spaces. Early training iterations resulted in a flat, globally average sheet that completely ignored the high-frequency plunge of the gravity well.
* **Solution:** Implemented a spatial **Distance Importance Weight Mask** in the loss function, scaling penalties up to 50x closer to the origin to force local convergence.

### 2. The Coordinate Singularity
Using pure polar coordinates introduces an unstable $1/r$ term at the origin, leading to training explosions or "donut holes."
* **Solution:** Engineered a **Multimodal Feature Injection** block $[x, y, r]$ where radial context is explicitly combined with Cartesian coordinates prior to the hidden layers, preserving circular symmetry without origin collapse.

---

## 📈 Performance & Results

By implementing an **Exponential Decay Learning Rate Schedule** via Optax to bounce out of local saddle-point plateaus, the engine achieved near-perfect convergence with an exceptionally tight error margin.

| Metric | Value |
| :--- | :--- |
| **Grid Resolution During Training** | 30 x 30 |
| **Training Steps** | 5,000 (Adam Optimizer) |
| **Mean Absolute Field Error (MAE)** | **0.012545** |

### Visualizing Convergence
* **Left:** Analytical True Field (Calculated via Newtonian Calculus).
* **Right:** PINN Predicted Field (Continuous Neural Inference).

*(Insert your final flawless V1 dashboard screenshot here!)*

---

## 🛠️ Tech Stack & Optimization
* **JAX:** High-performance numerical computing via XLA.
* **Equinox:** Clean, neural-network module filtering compatible with functional programming.
* **Optax:** Adaptive gradient tracking and schedule control.
* **Streamlit:** Interactive UI sandbox allowing parameter sweeping (Grid Resolution & $\epsilon$-softening) in real-time.
