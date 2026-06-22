# 🌌 PINN Space Engine: Gravitational Field Meta-Solver & Kinetic 
Accelerator

A Physics-Informed Neural Network (PINN) designed to solve complex 
multi-body gravitational potential fields and act as a real-time kinetic 
accelerator using **JAX**, **Equinox**, and **Optax**.

---

## 🚀 Milestone 1: The V1 Single-Body Baseline
Modeling a Newtonian point mass singularity ($U = -1/r$) introduces 
massive numerical hurdles:
* **The "Ghost Sheet" Problem (Spectral Bias):** Neural networks 
inherently prioritize low-frequency, smooth spaces, causing early 
iterations to completely ignore the sharp gravity well. *Solution: 
Implemented a spatial Distance Importance Weight Mask scaling penalties up 
to 50x near the origin.*
* **The Coordinate Singularity:** *Solution: Engineered a Multimodal 
Feature Injection block $[x, y, r]$ to preserve circular symmetry without 
origin collapse.*

---

## 🪐 Milestone 2: The V2 Multi-Body Meta-Solver
To scale the network from a toy problem to a generalized physics solver, 
the architecture was refactored to handle an arbitrary number of 
gravitational masses ($N$-bodies) without altering the core network 
layers.

### The Compression Math (Center-of-Mass Conditioning)
To feed an arbitrary number of moving bodies into a fixed-dimension neural 
network, the forward pass compresses the system configuration into 
invariant features:

1. **Translation Invariance:** Coordinates are evaluated relative to the 
system's shared Center of Mass (CoM):  
   $$x_{\text{scaled}} = x - x_{\text{com}}, \quad y_{\text{scaled}} = y - 
y_{\text{com}}$$

2. **Superposition Feature Injection:** The localized total gravitational 
signature is calculated and injected as an explicit radial pressure 
feature before the hidden layers:
   $$\text{net\_radial\_pull} = \sum_{i=1}^{N} \frac{M_i}{r_i}$$

---

## 🏎️ Milestone 3: The V3 Real-Time Orbit & Streamline Accelerator
Rather than acting as a passive field viewer, the architecture utilizes 
JAX automatic differentiation to serve as a high-speed vector physics 
engine.

### Exact Gradient Trajectory Tracking
In physics, gravitational acceleration is the negative gradient of the 
potential field:

$$\mathbf{a} = -\nabla U = -\left( \frac{\partial U}{\partial x}\hat{i} + 
\frac{\partial U}{\partial y}\hat{j} \right)$$

Instead of relying on unstable numerical grid approximations, the engine 
passes active particle arrays directly to the model using `jax.grad`. This 
computes exact force vectors on the fly, feeding an interactive kinetic 
solver:

$$\mathbf{v}_{t+1} = \mathbf{v}_t + \mathbf{a}_t \Delta t$$

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \mathbf{v}_t \Delta t$$

### Vector Calculus Fields (Dynamic Streamlines)
Using the same JAX-differentiated backend, the simulator maps the hidden 
force lines of the universe across the canvas using vector streamlines 
($\mathbf{g} = -\nabla U$). This reveals saddle points and gravitational 
balance vectors (Lagrange zones) visually in real-time.

---

## 🛸 Upcoming Milestone 4: Relativistic Photon Deflection
*(In Development)* Extending the PINN accelerator into General Relativity. 
By utilizing a modified Newtonian potential multiplier, the engine will 
simulate gravitational lensing, tracing how high-speed photon beams warp 
and split around deep spatial singularities.
