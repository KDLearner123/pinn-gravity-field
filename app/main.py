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

st.set_page_config(layout="wide")
st.title("🌌 PINN Space Engine: Multi-Body Spacetime Sandbox (V2)")
st.markdown("""
This sandbox uses a **Generalized Physics-Informed Neural Network (PINN)** to solve multi-body gravitational potential fields.
Rather than being fixed to one scenario, the model uses center-of-mass conditioning and mass-weighted radial features to solve arbitrary mass distributions.
""")

# 1. Setup UI Layout Sliders & Configuration Selector
st.sidebar.header("Simulation Parameters")
system_type = st.sidebar.selectbox(
    "Select Cosmic Configuration",
    ["Single Star", "Binary Star System", "Lagrange Three-Body"]
)
resolution = st.sidebar.slider("Grid Resolution", min_value=15, max_value=60, value=30)
softening = st.sidebar.slider("Singularity Softening (Epsilon)", min_value=0.05, max_value=0.5, value=0.1)

# 2. Map Selector to Body Struct Lists
if system_type == "Single Star":
    active_bodies = [{'pos': [0.0, 0.0], 'mass': 1.5}]
elif system_type == "Binary Star System":
    active_bodies = [
        {'pos': [-1.5, 0.0], 'mass': 1.0},
        {'pos': [1.5, 0.0],  'mass': 1.0}
    ]
else:  # Lagrange Three-Body
    active_bodies = [
        {'pos': [-2.0, -1.0], 'mass': 1.2},
        {'pos': [2.0, -1.0],  'mass': 1.2},
        {'pos': [0.0, 2.0],   'mass': 0.8}
    ]

# Construct static cache array keys out of configurations
body_key_string = "_".join([f"{b['pos']}_{b['mass']}" for b in active_bodies])
model_cache_key = f"v2_res_{resolution}_eps_{softening}_{hash(body_key_string)}"

# 3. Dynamic Training/Caching Block
if model_cache_key not in st.session_state:
    with st.spinner(f"Training Multi-Body PINN for {system_type}..."):
        st.session_state[model_cache_key] = train_pinn(
            steps=4000, 
            resolution=resolution, 
            epsilon=softening,
            body_list=active_bodies
        )
    st.success("Multi-Body PINN Engine Compiled and Trained Successfully!")

model = st.session_state[model_cache_key]

# 4. Generate Ground Truth & Model Evaluation Inputs
x = np.linspace(-5, 5, resolution)
y = np.linspace(-5, 5, resolution)
X, Y = np.meshgrid(x, y)
coords_np = np.stack([X.ravel(), Y.ravel()], axis=-1)
coords_jnp = jnp.array(coords_np)

body_positions_jnp = jnp.array([b['pos'] for b in active_bodies])
body_masses_jnp = jnp.array([b['mass'] for b in active_bodies])

# Analytical calculation via Superposition
true_u = np.zeros(coords_np.shape[0])
for b in active_bodies:
    r = np.sqrt((coords_np[:, 0] - b['pos'][0])**2 + (coords_np[:, 1] - b['pos'][1])**2 + softening**2)
    true_u += -b['mass'] / r

# Evaluate our updated PINN model across the field in a vectorized pass
predicted_u = jax.vmap(model, in_axes=(0, None, None))(coords_jnp, body_positions_jnp, body_masses_jnp)

# Reshape fields for spatial display
predicted_u = np.array(predicted_u).reshape(resolution, resolution)
true_u_grid = true_u.reshape(resolution, resolution)

# 5. Render Graphics
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot True Field
im0 = ax[0].imshow(true_u_grid, extent=[-5, 5, -5, 5], cmap="inferno", origin="lower", vmin=-4.0, vmax=-0.1)
ax[0].set_title(f"Analytical Multi-Body Field ({system_type})")
for b in active_bodies:
    ax[0].plot(b['pos'][0], b['pos'][1], 'wo', markersize=6) # Mark star positions
fig.colorbar(im0, ax=ax[0], label="Potential U")

# Plot PINN Model Field
im1 = ax[1].imshow(predicted_u, extent=[-5, 5, -5, 5], cmap="inferno", origin="lower", vmin=-4.0, vmax=-0.1)
ax[1].set_title("Generalized PINN Predicted Field")
for b in active_bodies:
    ax[1].plot(b['pos'][0], b['pos'][1], 'wo', markersize=6)
fig.colorbar(im1, ax=ax[1], label="Potential U")

plt.tight_layout()
st.pyplot(fig)

# Show performance metrics
absolute_error = np.mean(np.abs(true_u_grid - predicted_u))
st.metric(label="Mean Absolute Field Error (MAE)", value=f"{absolute_error:.6f}")