import streamlit as st
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import jax
import time

# Ensure app can find our source files natively
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import train_pinn
from src.physics import compute_acceleration_at_point

st.set_page_config(layout="wide", page_title="PINN Cosmic Sandbox Engine")

# Minimalist Production Header
st.title("🌌 PINN Cosmic Sandbox Engine")
st.caption("A fully continuous, mesh-free gravitational space simulator accelerated by a Physics-Informed Neural Network Meta-Solver.")

# 1. Sidebar Configurations
st.sidebar.header("System Controls")
system_type = st.sidebar.selectbox(
    "Select Universe Map",
    ["Single Star", "Binary Star System", "Lagrange Three-Body"]
)
resolution = st.sidebar.slider("Grid Fidelity", min_value=20, max_value=60, value=40)
softening = st.sidebar.slider("Singularity Softening", min_value=0.05, max_value=0.5, value=0.12)

st.sidebar.markdown("---")
st.sidebar.header("Engine Tuning")
matter_mode = st.sidebar.radio("Particle Profile Mode", ["Orbital Space Dust", "General Relativity Photon Beam"])
num_particles = st.sidebar.slider("Particle Count Density", min_value=20, max_value=200, value=120)
dt = st.sidebar.slider("Engine Time Step (dt)", min_value=0.01, max_value=0.10, value=0.04)
run_sim = st.sidebar.toggle("Engage Physics Engine", value=True)

# 2. Map Selector to Body Struct Lists
if system_type == "Single Star":
    active_bodies = [{'pos': [0.0, 0.0], 'mass': 2.0}]
elif system_type == "Binary Star System":
    active_bodies = [
        {'pos': [-1.6, 0.0], 'mass': 1.2},
        {'pos': [1.6, 0.0],  'mass': 1.2}
    ]
else:  # Lagrange Three-Body
    active_bodies = [
        {'pos': [-2.0, -1.0], 'mass': 1.5},
        {'pos': [2.0, -1.0],  'mass': 1.5},
        {'pos': [0.0, 2.2],   'mass': 1.0}
    ]

# Construct static cache array keys out of configurations
body_key_string = "_".join([f"{b['pos']}_{b['mass']}" for b in active_bodies])
model_cache_key = f"v4_res_{resolution}_eps_{softening}_{hash(body_key_string)}"

# 3. Dynamic Model Training/Caching Block
if model_cache_key not in st.session_state:
    with st.spinner("Compiling Neural Acceleration Vectors..."):
        st.session_state[model_cache_key] = train_pinn(
            steps=4000, 
            resolution=resolution, 
            epsilon=softening,
            body_list=active_bodies
        )

model = st.session_state[model_cache_key]

# Prepare JAX-ready static body variables
body_positions_jnp = jnp.array([b['pos'] for b in active_bodies])
body_masses_jnp = jnp.array([b['mass'] for b in active_bodies])

# 4. Generate the Fixed Background Neural Potential & Force Fields
x_grid = np.linspace(-5, 5, resolution)
y_grid = np.linspace(-5, 5, resolution)
X, Y = np.meshgrid(x_grid, y_grid)
coords_jnp = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))

predicted_u = jax.vmap(model, in_axes=(0, None, None))(coords_jnp, body_positions_jnp, body_masses_jnp)
predicted_u_grid = np.array(predicted_u).reshape(resolution, resolution)

# Compute acceleration vector streamlines
vectorized_acceleration = jax.vmap(compute_acceleration_at_point, in_axes=(None, 0, None, None))
grid_forces = vectorized_acceleration(model, coords_jnp, body_positions_jnp, body_masses_jnp)
fx_grid = np.array(grid_forces[:, 0]).reshape(resolution, resolution)
fy_grid = np.array(grid_forces[:, 1]).reshape(resolution, resolution)

# 5. Handle Particle Initializations based on Selected Mode
reset_triggered = st.sidebar.button("Reset Simulation Canvas")
mode_cache_key = f"current_mode_{system_type}"

if "p_pos" not in st.session_state or reset_triggered or st.session_state.get(mode_cache_key) != matter_mode:
    st.session_state[mode_cache_key] = matter_mode
    np.random.seed(42)
    
    if matter_mode == "Orbital Space Dust":
        # Standard circular orbit configuration setup
        angles = np.random.uniform(0, 2 * np.pi, num_particles)
        radii = np.random.uniform(2.0, 4.6, num_particles)
        st.session_state.p_pos = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)
        st.session_state.p_vel = np.stack([-0.55 * np.sin(angles), 0.55 * np.cos(angles)], axis=-1)
    else:
        # General Relativity Laser Photon Beam: Parallel horizontal tracks firing from the left
        y_tracks = np.linspace(-4.5, 4.5, num_particles)
        st.session_state.p_pos = np.stack([np.full(num_particles, -4.9), y_tracks], axis=-1)
        # Relativistic particles travel at a constant, uniform high speed
        st.session_state.p_vel = np.stack([np.full(num_particles, 2.5), np.zeros(num_particles)], axis=-1)

# 6. Full-Width Render Canvas Window Loop
plot_placeholder = st.empty()

while run_sim:
    pos = jnp.array(st.session_state.p_pos)
    vel = jnp.array(st.session_state.p_vel)
    
    accel = vectorized_acceleration(model, pos, body_positions_jnp, body_masses_jnp)
    
    # Under General Relativity theory, photons suffer twice the spatial trajectory deflection
    # compared to Newtonian mass bounds. We scale it elegantly if photon mode is engaged!
    multiplier = 2.0 if matter_mode == "General Relativity Photon Beam" else 1.0
    
    vel_new = vel + (accel * multiplier) * dt
    
    if matter_mode == "General Relativity Photon Beam":
        # Photons travel at a constant magnitude speed (speed of light proxy)
        speeds = jnp.sqrt(jnp.sum(vel_new**2, axis=1, keepdims=True)) + 1e-5
        vel_new = (vel_new / speeds) * 2.5
        
    pos_new = pos + vel_new * dt
    pos_np = np.array(pos_new)
    vel_np = np.array(vel_new)
    
    if matter_mode == "Orbital Space Dust":
        out_of_bounds = (np.abs(pos_np[:, 0]) > 4.9) | (np.abs(pos_np[:, 1]) > 4.9)
        vel_np[out_of_bounds] *= -0.7 
    else:
        # Respawn photon laser rays if they shoot out past the right boundary edge
        escaped = pos_np[:, 0] > 4.9
        if np.any(escaped):
            pos_np[escaped, 0] = -4.9
            pos_np[escaped, 1] = np.linspace(-4.5, 4.5, num_particles)[escaped]
            vel_np[escaped, 0] = 2.5
            vel_np[escaped, 1] = 0.0

    st.session_state.p_pos = pos_np
    st.session_state.p_vel = vel_np

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    ax.imshow(predicted_u_grid, extent=[-5, 5, -5, 5], cmap="inferno", origin="lower", alpha=0.9, vmin=-3.5, vmax=-0.1)
    ax.streamplot(X, Y, fx_grid, fy_grid, color=(1, 1, 1, 0.16), linewidth=0.8, density=1.2, arrowstyle='->')
    
    for b in active_bodies:
        ax.plot(b['pos'][0], b['pos'][1], marker='o', color='#ffffff', markersize=11, markeredgecolor='#ffaa00', mew=2)
        
    # Laser beams look amazing colored in radioactive hot magenta/cyan neon tones
    color_hex = '#ff007f' if matter_mode == "General Relativity Photon Beam" else '#00ffcc'
    ax.scatter(pos_np[:, 0], pos_np[:, 1], color=color_hex, s=14, alpha=0.95, edgecolors='none')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis('off')
    
    plot_placeholder.pyplot(fig)
    plt.close(fig)
    time.sleep(0.005)

# Static backup state rendering block
if not run_sim:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    ax.imshow(predicted_u_grid, extent=[-5, 5, -5, 5], cmap="inferno", origin="lower", alpha=0.9, vmin=-3.5, vmax=-0.1)
    ax.streamplot(X, Y, fx_grid, fy_grid, color=(1, 1, 1, 0.12), linewidth=0.8, density=1.2)
    for b in active_bodies:
        ax.plot(b['pos'][0], b['pos'][1], marker='o', color='#ffffff', markersize=11, markeredgecolor='#ffaa00', mew=2)
    color_hex = '#ff007f' if matter_mode == "General Relativity Photon Beam" else '#00ffcc'
    ax.scatter(st.session_state.p_pos[:, 0], st.session_state.p_pos[:, 1], color=color_hex, s=14, alpha=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis('off')
    plot_placeholder.pyplot(fig)
    plt.close(fig)