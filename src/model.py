import jax
import jax.numpy as jnp
import equinox as eqx

class PotentialPINN(eqx.Module):
    # Only structural weight layers belong as Module attributes
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 3)
        
        # FIX: Keep ONLY the Linear layers in this list
        self.layers = [
            eqx.nn.Linear(3, 64, key=keys[0]),
            eqx.nn.Linear(64, 64, key=keys[1]),
            eqx.nn.Linear(64, 1, key=keys[2])
        ]

    def __call__(self, coord):
        """
        Forward pass function.
        Transforms raw Cartesian inputs into Multimodal Features [x, y, r]
        and cleanly applies tanh activations between layers.
        """
        x = coord[0]
        y = coord[1]
        
        # 1. Compute radial distance from center
        r = jnp.sqrt(x**2 + y**2 + 1e-5)
        
        # 2. Package all 3 features together 
        in_features = jnp.array([x, y, r])
        
        # 3. Step sequentially through the layers, applying tanh manually
        val = in_features
        
        # Layer 1 -> Tanh
        val = jax.nn.tanh(self.layers[0](val))
        
        # Layer 2 -> Tanh
        val = jax.nn.tanh(self.layers[1](val))
        
        # Layer 3 (Output Layer - No Tanh so it can drop down to negative values!)
        val = self.layers[2](val)
        
        # Return a single scalar (Potential U)
        return val[0]