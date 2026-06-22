import streamlit as st
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import jax

# Ensure app can find our source files natively
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import train_pinn
from data.generate_data import generate_gravitational_field

st.set_page_config(layout="wide")
st.title("🌌 PINN Space Engine: Gravitational Field Meta-Solver")
st.markdown("""
This dashboard visualizes a **Physics-Informed Neural Network (PINN)** solving Laplace's Gravitational Equation ($\nabla^2 U = 0$). 
The network learns the underlying physical field rather than just memorizing a dataset.
""")

# 1. Setup UI Layout Sliders BEFORE the training block so the model can read them
st.sidebar.header("Simulation Parameters")
resolution = st.sidebar.slider("Grid Resolution", min_value=10, max_value=60, value=30)
softening = st.sidebar.slider("Singularity Softening (Epsilon)", min_value=0.05, max_value=0.5, value=0.1)

# Create a unique dynamic cache key based on slider settings
model_cache_key = f"model_res_{resolution}_eps_{softening}"

# 2. Dynamic Training/Caching Block
if model_cache_key not in st.session_state:
    with st.spinner("Training JAX PINN Engine for this configuration..."):
        # Pass slider parameters directly into your updated train_pinn function
        st.session_state[model_cache_key] = train_pinn(
            steps=5000, 
            resolution=resolution, 
            epsilon=softening
        )
    st.success("PINN Engine Compiled and Trained Successfully!")

model = st.session_state[model_cache_key]

# 3. Compute Predictions using the custom configured ground-truth data
coords, true_u = generate_gravitational_field(num_points=resolution, epsilon=softening)
coords_jnp = jnp.array(coords)

# Evaluate the model across the whole grid in one quick vectorized pass
predicted_u = jax.vmap(model)(coords_jnp)

# Convert from JAX array to standard NumPy and reshape into a 2D grid
predicted_u = np.array(predicted_u).reshape(resolution, resolution)
true_u_grid = true_u.reshape(resolution, resolution)

# 4. Plotting the Fields side-by-side using Matplotlib
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot True Gravitational Potential Field
im0 = ax[0].imshow(true_u_grid, extent=[-5, 5, -5, 5], cmap="inferno", origin="lower", vmin=-2.5, vmax=-0.1)
ax[0].set_title("Analytical True Field (Newtonian Mechanics)")
fig.colorbar(im0, ax=ax[0], label="Potential U")

# Plot Neural Network Predicted Field (using the exact same color limits!)
im1 = ax[1].imshow(predicted_u, extent=[-5, 5, -5, 5], cmap="inferno", origin="lower", vmin=-2.5, vmax=-0.1)
ax[1].set_title("PINN Predicted Field (AI Solver)")
fig.colorbar(im1, ax=ax[1], label="Potential U")

# Clean layout and render straight into the Streamlit web layout
plt.tight_layout()
st.pyplot(fig)

# Show error metric to prove accuracy
absolute_error = np.mean(np.abs(true_u_grid - predicted_u))
st.metric(label="Mean Absolute Field Error", value=f"{absolute_error:.6f}")