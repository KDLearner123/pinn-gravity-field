import numpy as np
def generate_gravitational_field(num_points=100, low=-5.0, high=5.0, epsilon=0.1):
    """
    Generates a 2D grid of coordinates and calculates the analytical 
    gravitational potential (U) at each point.
    """
    # 1. Create a 1D array of evenly spaced points for x and y axes
    x = np.linspace(low, high, num_points)
    y = np.linspace(low, high, num_points)
    
    # 2. Create a 2D grid (Meshgrid) from the 1D arrays
    X, Y = np.meshgrid(x, y)
    
    # 3. Calculate the distance 'r' from the origin (0,0) with softening
    r = np.sqrt(X**2 + Y**2 + epsilon)
    
    # 4. Compute the analytical potential U = -1 / r
    U = -1.0 / r
    
    # 5. Reshape data into flat columns for machine learning input/output
    # Coordinates shape: (N, 2) -> columns of [x, y]
    coordinates = np.stack([X.ravel(), Y.ravel()], axis=1)
    # Potentials shape: (N, 1) -> column of [U]
    potentials = U.ravel().reshape(-1, 1)
    
    return coordinates, potentials

if __name__ == "__main__":
    # Test our generator
    coords, potentials = generate_gravitational_field(num_points=5)
    print("Sample Coordinates (x, y):\n", coords[:5])
    print("\nCorresponding True Potentials (U):\n", potentials[:5])
