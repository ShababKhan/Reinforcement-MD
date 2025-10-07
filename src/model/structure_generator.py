"""
Structure generation utilities for the rMD system.
"""
import numpy as np
import torch
from scipy import interpolate


class StructureGenerator:
    """
    Generator for protein structures using the trained rMD autoencoder.
    """
    
    def __init__(self, model, free_energy_map=None, device=None):
        """
        Initialize the structure generator.
        
        Parameters:
        -----------
        model : InformedAutoencoder
            The trained autoencoder model.
        free_energy_map : object, optional
            Free energy map for checking energy at sampled points.
            Should implement a get_value_at_point(point) method.
        device : torch.device, optional
            The device to use for inference.
            If None, will use CUDA if available.
        """
        self.model = model
        self.free_energy_map = free_energy_map
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model.to(self.device)
        self.model.eval()
    
    def generate_from_cv_coordinates(self, cv_coordinates):
        """
        Generate protein structures from CV coordinates.
        
        Parameters:
        -----------
        cv_coordinates : numpy.ndarray
            CV coordinates with shape (n_points, cv_dim).
            
        Returns:
        --------
        numpy.ndarray
            Generated protein coordinates with shape (n_points, n_atoms * 3).
        """
        # Convert to torch tensor
        cv_tensor = torch.tensor(cv_coordinates, dtype=torch.float32).to(self.device)
        
        # Generate structures
        with torch.no_grad():
            generated = self.model.decode(cv_tensor)
            
        # Convert to numpy
        return generated.cpu().numpy()
    
    def generate_along_path(self, path_points, n_samples=20):
        """
        Generate protein structures along a path in CV space.
        
        Parameters:
        -----------
        path_points : numpy.ndarray
            Points defining the path in CV space with shape (n_points, cv_dim).
        n_samples : int, optional
            Number of structures to generate along the path.
            
        Returns:
        --------
        tuple
            Tuple containing:
            - structures: Generated protein coordinates with shape (n_samples, n_atoms * 3).
            - path_points: Sampled points along the path with shape (n_samples, cv_dim).
            - energies: Free energy values at each point if free_energy_map is provided.
        """
        # Sample points along the path
        t = np.linspace(0, 1, len(path_points))
        t_interp = np.linspace(0, 1, n_samples)
        
        # Interpolate each dimension
        cv_dim = path_points.shape[1]
        sampled_points = np.zeros((n_samples, cv_dim))
        
        for i in range(cv_dim):
            # Use cubic spline interpolation for smooth paths
            spline = interpolate.splrep(t, path_points[:, i], k=min(3, len(path_points)-1))
            sampled_points[:, i] = interpolate.splev(t_interp, spline)
        
        # Generate structures
        structures = self.generate_from_cv_coordinates(sampled_points)
        
        # Compute free energies if map is provided
        energies = None
        if self.free_energy_map is not None:
            energies = np.zeros(n_samples)
            for i in range(n_samples):
                energies[i] = self.free_energy_map.get_value_at_point(sampled_points[i])
        
        return structures, sampled_points, energies
    
    def generate_b_spline_path(self, anchor_points, n_samples=20):
        """
        Generate a B-spline path through anchor points in CV space.
        
        Parameters:
        -----------
        anchor_points : numpy.ndarray
            Anchor points in CV space with shape (n_anchors, cv_dim).
        n_samples : int, optional
            Number of structures to generate along the path.
            
        Returns:
        --------
        tuple
            Tuple containing:
            - structures: Generated protein coordinates with shape (n_samples, n_atoms * 3).
            - path_points: Sampled points along the path with shape (n_samples, cv_dim).
            - energies: Free energy values at each point if free_energy_map is provided.
        """
        # Create a B-spline through the anchor points
        n_anchors = len(anchor_points)
        cv_dim = anchor_points.shape[1]
        
        # Parameter space for the B-spline
        t = np.linspace(0, 1, n_anchors)
        
        # Create B-spline representation
        path_points = np.zeros((n_samples, cv_dim))
        t_interp = np.linspace(0, 1, n_samples)
        
        for i in range(cv_dim):
            # Use B-spline interpolation
            spline = interpolate.splrep(t, anchor_points[:, i], k=min(3, n_anchors-1))
            path_points[:, i] = interpolate.splev(t_interp, spline)
        
        # Generate structures along the path
        return self.generate_along_path(path_points, n_samples)
    
    def find_low_energy_path(self, start_point, end_point, n_samples=20, n_paths=5):
        """
        Find a low free-energy path between two points in CV space.
        
        Parameters:
        -----------
        start_point : numpy.ndarray
            Starting point in CV space with shape (cv_dim,).
        end_point : numpy.ndarray
            End point in CV space with shape (cv_dim,).
        n_samples : int, optional
            Number of structures to generate along each candidate path.
        n_paths : int, optional
            Number of candidate paths to evaluate.
            
        Returns:
        --------
        tuple
            Tuple containing:
            - best_structures: Generated protein coordinates along the best path.
            - best_path_points: Sampled points along the best path.
            - best_energies: Free energy values along the best path.
        """
        if self.free_energy_map is None:
            raise ValueError("Free energy map required for finding low-energy paths")
        
        best_path = None
        best_path_energy = float('inf')
        best_structures = None
        best_path_points = None
        best_energies = None
        
        # Create multiple candidate paths with different intermediate points
        for i in range(n_paths):
            # For simplicity, we'll use random intermediate points
            # In a real implementation, more sophisticated path planning could be used
            cv_dim = len(start_point)
            
            # Random number of intermediate points (1-3)
            n_intermediate = np.random.randint(1, 4)
            
            # Create anchor points: start, intermediates, end
            anchor_points = np.zeros((n_intermediate + 2, cv_dim))
            anchor_points[0] = start_point
            anchor_points[-1] = end_point
            
            # Generate random intermediate points along a perturbed direct path
            direct_path = end_point - start_point
            for j in range(n_intermediate):
                t = (j + 1) / (n_intermediate + 1)
                # Add some random perturbation to the direct path
                perturbation = np.random.normal(0, 0.1, cv_dim)
                anchor_points[j+1] = start_point + t * direct_path + perturbation
            
            # Generate structures along this path
            structures, path_points, energies = self.generate_b_spline_path(anchor_points, n_samples)
            
            # Calculate total path energy (max energy or integral)
            path_energy = np.max(energies)  # Or use np.trapz(energies) for integral
            
            # Check if this path is better
            if path_energy < best_path_energy:
                best_path_energy = path_energy
                best_structures = structures
                best_path_points = path_points
                best_energies = energies
        
        return best_structures, best_path_points, best_energies
    
    def explore_conformational_space(self, center_point, radius, n_points=100):
        """
        Explore conformational space around a center point in CV space.
        
        Parameters:
        -----------
        center_point : numpy.ndarray
            Center point in CV space with shape (cv_dim,).
        radius : float
            Radius around the center point to explore.
        n_points : int, optional
            Number of points to sample in the exploration space.
            
        Returns:
        --------
        tuple
            Tuple containing:
            - structures: Generated protein coordinates with shape (n_points, n_atoms * 3).
            - sampled_points: Sampled points in CV space with shape (n_points, cv_dim).
            - energies: Free energy values at each point if free_energy_map is provided.
        """
        cv_dim = len(center_point)
        
        # Generate random points within a hypersphere around the center
        # First generate random directions
        random_directions = np.random.randn(n_points, cv_dim)
        # Normalize to unit vectors
        random_directions /= np.linalg.norm(random_directions, axis=1)[:, np.newaxis]
        
        # Generate random radii (use r^(1/dim) distribution for uniform sampling in hypersphere)
        random_radii = radius * np.random.random(n_points) ** (1.0 / cv_dim)
        
        # Scale directions by radii
        offsets = random_directions * random_radii[:, np.newaxis]
        
        # Add center point
        sampled_points = center_point + offsets
        
        # Generate structures
        structures = self.generate_from_cv_coordinates(sampled_points)
        
        # Compute free energies if map is provided
        energies = None
        if self.free_energy_map is not None:
            energies = np.zeros(n_points)
            for i in range(n_points):
                energies[i] = self.free_energy_map.get_value_at_point(sampled_points[i])
        
        return structures, sampled_points, energies