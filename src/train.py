import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from src.model import PotentialPINN
from src.physics import compute_total_loss

def train_pinn(steps=4000, resolution=30, epsilon=0.1, body_list=None):
    key = jax.random.PRNGKey(42)
    model = PotentialPINN(key)
    
    # 1. Setup multi-body configurations
    if body_list is None:
        # Default to binary system coordinates
        body_list = [
            {'pos': [-1.5, 0.0], 'mass': 1.0},
            {'pos': [1.5, 0.0],  'mass': 1.0}
        ]
        
    body_positions = jnp.array([b['pos'] for b in body_list])
    body_masses = jnp.array([b['mass'] for b in body_list])
    
    # 2. Build spatial environment grid
    x = jnp.linspace(-5, 5, resolution)
    y = jnp.linspace(-5, 5, resolution)
    X, Y = jnp.meshgrid(x, y)
    batch_coords = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    
    # Calculate True field via Superposition
    batch_true_u = jnp.zeros(batch_coords.shape[0])
    for b in body_list:
        bx, by = b['pos']
        r = jnp.sqrt((batch_coords[:, 0] - bx)**2 + (batch_coords[:, 1] - by)**2 + epsilon**2)
        batch_true_u += -b['mass'] / r
        
    margin = 4.99
    boundary_mask = (jnp.abs(batch_coords[:, 0]) > margin) | (jnp.abs(batch_coords[:, 1]) > margin)
    boundary_indices = jnp.where(boundary_mask)[0]
    
    # 3. Optimization Setup
    lr_schedule = optax.exponential_decay(
        init_value=1e-3, transition_steps=1000, decay_rate=0.9, staircase=False
    )
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def make_step(model, opt_state, coords, true_u, b_indices, b_pos, b_mass):
        def loss_fn(m):
            return compute_total_loss(m, coords, true_u, b_indices, b_pos, b_mass)
        loss, grads = jax.value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print(f"Training Multi-Body Solver ({len(body_list)} Bodies Active)...")
    for step in range(steps + 1):
        model, opt_state, loss = make_step(
            model, opt_state, batch_coords, batch_true_u, boundary_indices, body_positions, body_masses
        )
        if step % 500 == 0:
            print(f"Step {step:5d} | Multi-Body Composite Loss: {loss:.6f}")
            
    return model