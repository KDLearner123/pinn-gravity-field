import jax
import jax.numpy as jnp
import equinox as eqx

class PotentialPINN(eqx.Module):
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 3)
        
        # We process the localized spatial/gravitational features
        # Input dim is 3: [x_center_of_mass, y_center_of_mass, effective_radial_pull]
        self.layers = [
            eqx.nn.Linear(3, 64, key=keys[0]),
            eqx.nn.Linear(64, 64, key=keys[1]),
            eqx.nn.Linear(64, 1, key=keys[2])
        ]

    def __call__(self, coord, body_positions, body_masses):
        """
        Forward pass evaluated at a single probe coordinate [x, y],
        conditioned on an arbitrary matrix of body configurations.
        """
        x, y = coord[0], coord[1]
        
        # 1. Compute relative vectors to all bodies simultaneously using broadcasting
        dx = x - body_positions[:, 0]
        dy = y - body_positions[:, 1]
        r_sq = dx**2 + dy**2 + 1e-5
        r = jnp.sqrt(r_sq)
        
        # 2. Extract mass-weighted features (Center of Mass influences & Net Radial Pressure)
        # This collapses an arbitrary number of bodies N into static feature dimensions!
        total_mass = jnp.sum(body_masses) + 1e-5
        x_com = jnp.sum(body_masses * body_positions[:, 0]) / total_mass
        y_com = jnp.sum(body_masses * body_positions[:, 1]) / total_mass
        
        # Effective localized gravitational warp scalar
        net_radial_pull = jnp.sum(body_masses / r)
        
        # 3. Formulate the dynamic multimodal feature array
        in_features = jnp.array([x - x_com, y - y_com, net_radial_pull])
        
        # 4. Forward execution through hidden network
        val = jnp.tanh(self.layers[0](in_features))
        val = jnp.tanh(self.layers[1](val))
        val = self.layers[2](val)
        
        return val[0]