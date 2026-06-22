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

st.title("🌌 PINN Cosmic Sandbox Engine")
st.caption("A fully dynamic, mesh-free gravitational space simulator accelerated by a Physics-Informed Neural Network Meta-Solver.")

# 1. Sidebar Configurations
st.sidebar.header("System Controls")
system_type = st.sidebar.selectbox(
    "Select Universe Map",
    ["Single Star", "Binary Star System", "Lagrange Three-Body"]
)
resolution = st.sidebar.slider("Grid Fidelity", min_value=20, max_value=60, value=35)
softening = st.sidebar.slider("Singularity Softening", min_value=0.05, max_value=0.5, value=0.12)

st.sidebar.markdown("---")
st.sidebar.header("Engine Tuning")
matter_mode = st.sidebar.radio("Particle Profile Mode", ["Orbital Space Dust", "General Relativity Photon Beam"])
num_particles = st.sidebar.slider("Particle Count Density", min_value=20, max_value=200, value=100)
dt = st.sidebar.slider("Engine Time Step (dt)", min_value=0.01, max_value=0.10, value=0.03)

st.sidebar.markdown("---")
st.sidebar.header("Advanced Physics")
move_stars = st.sidebar.toggle("Enable Star Orbital Dynamics", value=True)
run_sim = st.sidebar.toggle("Engage Physics Engine", value=True)

# 2. Map Selector to Body Struct Lists with seeded velocities for N-body orbits
if system_type == "Single Star":
    initial_bodies = [{'pos': [0.0, 0.0], 'vel': [0.0, 0.0], 'mass': 2.0}]
elif system_type == "Binary Star System":
    # Balanced stable circular velocities for equal masses
    initial_bodies = [
        {'pos': [-1.6, 0.0], 'vel': [0.0, -0.45], 'mass': 1.5},
        {'pos': [1.6, 0.0],  'vel': [0.0, 0.45],  'mass': 1.5}
    ]
else:  # Lagrange Three-Body
    # Seeded circular trajectory vector adjustments
    initial_bodies = [
        {'pos': [-2.0, -1.0], 'vel': [0.2, -0.35], 'mass': 1.5},
        {'pos': [2.0, -1.0],  'vel': [-0.2, 0.35], 'mass': 1.5},
        {'pos': [0.0, 2.2],   'vel': [0.4, 0.0],   'mass': 1.0}
    ]

# Construct unique static cache key for model architecture
body_key_string = "_".join([f"{b['mass']}" for b in initial_bodies])
model_cache_key = f"v5_res_{resolution}_eps_{softening}_{hash(body_key_string)}"

if model_cache_key not in st.session_state:
    with st.spinner("Compiling Neural Acceleration Vectors..."):
        st.session_state[model_cache_key] = train_pinn(
            steps=4000, 
            resolution=resolution, 
            epsilon=softening,
            body_list=initial_bodies
        )

model = st.session_state[model_cache_key]

# 3. Handle Persistent State Tracking (Particles & Massive Stars)
reset_triggered = st.sidebar.button("Reset Simulation Canvas")
mode_cache_key = f"current_mode_{system_type}"

if "p_pos" not in st.session_state or reset_triggered or st.session_state.get(mode_cache_key) != matter_mode:
    st.session_state[mode_cache_key] = matter_mode
    np.random.seed(42)
    
    # Initialize Star State Vectors
    st.session_state.s_pos = np.array([b['pos'] for b in initial_bodies])
    st.session_state.s_vel = np.array([b['vel'] for b in initial_bodies])
    st.session_state.s_mass = np.array([b['mass'] for b in initial_bodies])
    
    if matter_mode == "Orbital Space Dust":
        angles = np.random.uniform(0, 2 * np.pi, num_particles)
        radii = np.random.uniform(2.0, 4.6, num_particles)
        st.session_state.p_pos = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)
        st.session_state.p_vel = np.stack([-0.55 * np.sin(angles), 0.55 * np.cos(angles)], axis=-1)
    else:
        y_tracks = np.linspace(-4.5, 4.5, num_particles)
        st.session_state.p_pos = np.stack([np.full(num_particles, -4.9), y_tracks], axis=-1)
        st.session_state.p_vel = np.stack([np.full(num_particles, 2.5), np.zeros(num_particles)], axis=-1)

vectorized_acceleration = jax.vmap(compute_acceleration_at_point, in_axes=(None, 0, None, None))

# 4. Main Simulation Loop
plot_placeholder = st.empty()

while run_sim:
    # Read active states
    s_pos = st.session_state.s_pos
    s_vel = st.session_state.s_vel
    s_mass = st.session_state.s_mass
    
    p_pos = jnp.array(st.session_state.p_pos)
    p_vel = jnp.array(st.session_state.p_vel)
    
    body_positions_jnp = jnp.array(s_pos)
    body_masses_jnp = jnp.array(s_mass)
    
    # STEP A: Classical N-Body Calculations for the Stars themselves
    if move_stars and len(s_mass) > 1:
        s_accel = np.zeros_like(s_pos)
        for i in range(len(s_pos)):
            for j in range(len(s_pos)):
                if i != j:
                    r_vec = s_pos[j] - s_pos[i]
                    dist = np.linalg.norm(r_vec) + softening # use softening to prevent division by zero explode
                    # Newton's law: acceleration = G * M_j * r_vec / dist^3
                    s_accel[i] += s_mass[j] * r_vec / (dist**3)
        
        s_vel += s_accel * dt
        s_pos += s_vel * dt
        
        # Keep stars bound near center scene mapping
        outside = np.linalg.norm(s_pos, axis=1) > 4.5
        if np.any(outside):
            s_vel[outside] *= -0.9
            
        st.session_state.s_pos = s_pos
        st.session_state.s_vel = s_vel

    # STEP B: PINN Accelerated Calculations for Dust / Photons
    accel = vectorized_acceleration(model, p_pos, body_positions_jnp, body_masses_jnp)
    multiplier = 2.0 if matter_mode == "General Relativity Photon Beam" else 1.0
    
    p_vel_new = p_vel + (accel * multiplier) * dt
    
    if matter_mode == "General Relativity Photon Beam":
        speeds = jnp.sqrt(jnp.sum(p_vel_new**2, axis=1, keepdims=True)) + 1e-5
        p_vel_new = (p_vel_new / speeds) * 2.5
        
    p_pos_new = p_pos + p_vel_new * dt
    p_pos_np = np.array(p_pos_new)
    p_vel_np = np.array(p_vel_new)
    
    if matter_mode == "Orbital Space Dust":
        out_of_bounds = (np.abs(p_pos_np[:, 0]) > 4.9) | (np.abs(p_pos_np[:, 1]) > 4.9)
        p_vel_np[out_of_bounds] *= -0.7 
    else:
        escaped = p_pos_np[:, 0] > 4.9
        if np.any(escaped):
            p_pos_np[escaped, 0] = -4.9
            p_pos_np[escaped, 1] = np.linspace(-4.5, 4.5, num_particles)[escaped]
            p_vel_np[escaped, 0] = 2.5
            p_vel_np[escaped, 1] = 0.0

    st.session_state.p_pos = p_pos_np
    st.session_state.p_vel = p_vel_np

    # STEP C: Dynamic Mesh Background Grid Generation for Streamlines
    x_grid = np.linspace(-5, 5, resolution)
    y_grid = np.linspace(-5, 5, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    coords_jnp = jnp.array(np.stack([X.ravel(), Y.ravel()], axis=-1))
    
    predicted_u = jax.vmap(model, in_axes=(0, None, None))(coords_jnp, body_positions_jnp, body_masses_jnp)
    predicted_u_grid = np.array(predicted_u).reshape(resolution, resolution)
    
    grid_forces = vectorized_acceleration(model, coords_jnp, body_positions_jnp, body_masses_jnp)
    fx_grid = np.array(grid_forces[:, 0]).reshape(resolution, resolution)
    fy_grid = np.array(grid_forces[:, 1]).reshape(resolution, resolution)

    # STEP D: Render Frame Graphics
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    ax.imshow(predicted_u_grid, extent=[-5, 5, -5, 5], cmap="inferno", origin="lower", alpha=0.9, vmin=-3.5, vmax=-0.1)
    ax.streamplot(X, Y, fx_grid, fy_grid, color=(1, 1, 1, 0.16), linewidth=0.8, density=1.2, arrowstyle='->')
    
    # Draw moving stars
    for pos in s_pos:
        ax.plot(pos[0], pos[1], marker='o', color='#ffffff', markersize=12, markeredgecolor='#ffaa00', mew=2.5)
        
    color_hex = '#ff007f' if matter_mode == "General Relativity Photon Beam" else '#00ffcc'
    ax.scatter(p_pos_np[:, 0], p_pos_np[:, 1], color=color_hex, s=14, alpha=0.95, edgecolors='none')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axis('off')
    
    plot_placeholder.pyplot(fig)
    plt.close(fig)
    time.sleep(0.005)