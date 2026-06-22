import jax
import jax.numpy as jnp

def laplace_residual(model, coord, body_positions, body_masses):
    """
    Computes Laplace PDE violation for multi-body space.
    """
    # Isolate grad with respect to coordinate argument only (argnums=0)
    grad_u = jax.grad(model, argnums=0)
    
    def du_dx(c): return grad_u(c, body_positions, body_masses)[0]
    def du_dy(c): return grad_u(c, body_positions, body_masses)[1]
    
    d2u_dx2 = jax.grad(du_dx)(coord)[0]
    d2u_dy2 = jax.grad(du_dy)(coord)[1]
    
    return jnp.square(d2u_dx2 + d2u_dy2)

def compute_total_loss(model, batch_coords, batch_true_u, boundary_indices, body_positions, body_masses):
    """
    Evaluates multi-body PINN performance across the spatial field configuration.
    """
    # 1. Vectorized evaluation over coordinates while locking body metadata static
    vectorized_physics = jax.vmap(laplace_residual, in_axes=(None, 0, None, None))
    physics_violations = vectorized_physics(model, batch_coords, body_positions, body_masses)
    mean_physics_loss = jnp.mean(physics_violations)
    
    # 2. Evaluate model predictions
    vectorized_model = jax.vmap(model, in_axes=(0, None, None))
    predicted_potentials = vectorized_model(batch_coords, body_positions, body_masses)
    
    # 3. Dynamic Importance Weight Masking 
    # Finds proximity to *any* mass singularity point in the landscape
    min_distances = jnp.min(
        jnp.sqrt(
            jnp.square(batch_coords[:, None, 0] - body_positions[None, :, 0]) +
            jnp.square(batch_coords[:, None, 1] - body_positions[None, :, 1]) + 1e-5
        ), axis=1
    )
    importance_weights = 1.0 + (50.0 / (min_distances + 0.5))
    
    # 4. Weighted Data & Boundary Loss calculation
    squared_errors = jnp.square(predicted_potentials - batch_true_u)
    weighted_data_loss = jnp.mean(squared_errors * importance_weights)
    
    pred_boundary = predicted_potentials[boundary_indices]
    true_boundary = batch_true_u[boundary_indices]
    mean_boundary_loss = jnp.mean(jnp.square(pred_boundary - true_boundary))
    
    return 10.0 * (mean_boundary_loss + weighted_data_loss) + 0.01 * mean_physics_loss