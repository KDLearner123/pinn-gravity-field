import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from src.model import PotentialPINN
from src.physics import compute_total_loss

def train_pinn(steps=3000, resolution=30, epsilon=0.1):
    
    # 1. Setup Random Keys for Equinox Model Initialization
    key = jax.random.PRNGKey(42)
    model_key, data_key = jax.random.split(key)
    
    # Initialize the model (Ensure input dim matches your model.py setup!)
    model = PotentialPINN(model_key)
    
    # 2. Generate/Load Synthetic Training Data
    # Creating a 30x30 spatial grid
    res = resolution
    x = jnp.linspace(-5, 5, res)
    y = jnp.linspace(-5, 5, res)
    X, Y = jnp.meshgrid(x, y)
    batch_coords = jnp.stack([X.ravel(), Y.ravel()], axis=-1) # Shape: (900, 2)
    
    # Analytical True Field calculation (with small epsilon softening)
    eps = epsilon
    r = jnp.sqrt(batch_coords[:, 0]**2 + batch_coords[:, 1]**2 + eps**2)
    batch_true_u = -1.0 / r
    
    # Identify Boundary Indices (where x or y hit the edges -5 or 5)
    margin = 4.99
    boundary_mask = (jnp.abs(batch_coords[:, 0]) > margin) | (jnp.abs(batch_coords[:, 1]) > margin)
    boundary_indices = jnp.where(boundary_mask)[0]

    # 3. Define the Dynamic Learning Rate Schedule
    # Starts aggressive at 1e-3, then slowly decays to fine-tune the center well
    lr_schedule = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=1000,
        decay_rate=0.9,
        staircase=False
    )
    
    # 4. Setup Optimizer
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # 5. Define a Pure, JIT-compilable Step Function
    @eqx.filter_jit
    def make_step(model, opt_state, coords, true_u, b_indices):
        def loss_fn(runnable_model):
            return compute_total_loss(runnable_model, coords, true_u, b_indices)
        
        loss, grads = jax.value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # 6. Main Training Loop
    print("Beginning Training Loop...")
    for step in range(steps + 1):
        model, opt_state, loss = make_step(
            model, opt_state, batch_coords, batch_true_u, boundary_indices
        )
        
        # Print progress every 200 steps to the terminal
        if step % 200 == 0:
            print(f"Step {step:5d} | Total Composite Loss: {loss:.6f}")
            
    return model