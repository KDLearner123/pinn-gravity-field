import numpy as np

def generate_multi_body_field(num_points=30, epsilon=0.1, bodies=None):
    """
    Generates a gravitational potential field for an arbitrary number of masses.
    
    bodies: List of dictionaries containing {'pos': [x, y], 'mass': m}
    """
    # Default to a binary star system if no bodies are specified
    if bodies is None:
        bodies = [
            {'pos': [-1.5, 0.0], 'mass': 1.0}, # Star A
            {'pos': [1.5, 0.0],  'mass': 1.0}  # Star B
        ]
        
    x = np.linspace(-5, 5, num_points)
    y = np.linspace(-5, 5, num_points)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid to match our training coordinate shape (N_points, 2)
    coords = np.stack([X.ravel(), Y.ravel()], axis=-1)
    
    # Initialize total potential field to zero
    true_u = np.zeros(coords.shape[0])
    
    # Compute superposition of potentials
    for body in bodies:
        bx, by = body['pos']
        m = body['mass']
        r = np.sqrt((coords[:, 0] - bx)**2 + (coords[:, 1] - by)**2 + epsilon**2)
        true_u += -m / r
        
    return coords, true_u, bodies