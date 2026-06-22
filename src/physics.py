import jax
import jax.numpy as jnp

def laplace_residual(model, coord):
    """
    Computes the standard Cartesian Laplace PDE violation using JAX Autodiff:
    Residual = d^2U/dx^2 + d^2U/dy^2
    """
    # Define a function to get gradients of U with respect to coordinates
    grad_u = jax.grad(model)
    
    # Define functions to isolate single derivative elements for the Hessian
    def du_dx(c): return grad_u(c)[0]
    def du_dy(c): return grad_u(c)[1]
    
    # Differentiate a second time to get second derivatives
    d2u_dx2 = jax.grad(du_dx)(coord)[0]
    d2u_dy2 = jax.grad(du_dy)(coord)[1]
    
    # In empty space, Laplace states these must sum to 0
    return jnp.square(d2u_dx2 + d2u_dy2)

def compute_total_loss(model, batch_coords, batch_true_u, boundary_indices):
    """
    Combines the physics constraint (Laplace) with an importance-weighted 
    interior data loss to pull the network into the deep center well.
    """
    # 1. Calculate physics residual across all points using jax.vmap
    vectorized_physics_loss = jax.vmap(laplace_residual, in_axes=(None, 0))
    physics_violations = vectorized_physics_loss(model, batch_coords)
    mean_physics_loss = jnp.mean(physics_violations)
    
    # 2. Evaluate model predictions across the whole batch
    predicted_potentials = jax.vmap(model)(batch_coords)
    
    # 3. Calculate spatial distance mask to combat sample averaging laziness
    distances = jnp.sqrt(batch_coords[:, 0]**2 + batch_coords[:, 1]**2 + 1e-5)
    
    # Points near the center scale up to a 50x penalty multiplier; edges stay at 1x
    importance_weights = 1.0 + (50.0 / (distances + 0.5))
    
    # 4. Compute Weighted Data Loss
    squared_errors = jnp.square(predicted_potentials - batch_true_u)
    weighted_data_loss = jnp.mean(squared_errors * importance_weights)
    
    # 5. Compute Boundary Loss (Enforces the outer container values match)
    pred_boundary = predicted_potentials[boundary_indices]
    true_boundary = batch_true_u[boundary_indices]
    mean_boundary_loss = jnp.mean(jnp.square(pred_boundary - true_boundary))
    
    # Composite balance: High priority to data anchors, light smoothing from physics
    return 10.0 * (mean_boundary_loss + weighted_data_loss) + 0.01 * mean_physics_loss