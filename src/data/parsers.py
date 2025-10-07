"""
Parsers for molecular dynamics file formats.
"""
import numpy as np


class PDBParser:
    """
    Parser for Protein Data Bank (PDB) file format.
    """
    
    def __init__(self):
        """
        Initialize the PDB parser.
        """
        self.atom_records = []
        self.coordinates = None
        self.atom_names = []
        self.residue_names = []
        self.chain_ids = []
        self.residue_ids = []
        self.element_symbols = []
        
    def parse_file(self, file_path):
        """
        Parse a PDB file and extract atomic coordinates and metadata.
        
        Parameters:
        -----------
        file_path : str
            Path to the PDB file.
            
        Returns:
        --------
        dict
            Dictionary containing atomic coordinates and metadata.
        """
        self.atom_records = []
        self.atom_names = []
        self.residue_names = []
        self.chain_ids = []
        self.residue_ids = []
        self.element_symbols = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    self.atom_records.append(line.strip())
                    
                    # Extract atom information
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21:22].strip()
                    residue_id = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    
                    # Extract element symbol if available
                    if len(line) >= 78:
                        element = line[76:78].strip()
                    else:
                        # Infer element from atom name
                        element = atom_name[0]
                        if len(atom_name) > 1 and not atom_name[1].isdigit():
                            if atom_name[0:2] in ['BR', 'CL', 'ZN', 'FE', 'MG']:
                                element = atom_name[0:2]
                    
                    self.atom_names.append(atom_name)
                    self.residue_names.append(residue_name)
                    self.chain_ids.append(chain_id)
                    self.residue_ids.append(residue_id)
                    self.element_symbols.append(element)
        
        # Convert coordinates to NumPy array
        num_atoms = len(self.atom_records)
        self.coordinates = np.zeros((num_atoms, 3))
        
        for i, line in enumerate(self.atom_records):
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            self.coordinates[i] = [x, y, z]
            
        return {
            'coordinates': self.coordinates,
            'atom_names': self.atom_names,
            'residue_names': self.residue_names,
            'chain_ids': self.chain_ids,
            'residue_ids': self.residue_ids,
            'element_symbols': self.element_symbols
        }
    
    def get_coordinates(self):
        """
        Get the atomic coordinates.
        
        Returns:
        --------
        numpy.ndarray
            Array of atomic coordinates with shape (n_atoms, 3).
        """
        return self.coordinates


class TrajectoryParser:
    """
    Parser for molecular dynamics trajectory files.
    This is a base class that should be extended for specific trajectory formats.
    """
    
    def __init__(self):
        """
        Initialize the trajectory parser.
        """
        self.n_frames = 0
        self.n_atoms = 0
        self.coordinates = None
        self.time = None
        
    def parse_file(self, file_path):
        """
        Parse a trajectory file and extract coordinates for each frame.
        
        Parameters:
        -----------
        file_path : str
            Path to the trajectory file.
            
        Returns:
        --------
        dict
            Dictionary containing trajectory data.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def get_frame(self, frame_index):
        """
        Get coordinates for a specific frame.
        
        Parameters:
        -----------
        frame_index : int
            Index of the desired frame.
            
        Returns:
        --------
        numpy.ndarray
            Array of atomic coordinates for the specified frame with shape (n_atoms, 3).
        """
        if self.coordinates is None:
            raise ValueError("No trajectory loaded")
        
        if frame_index < 0 or frame_index >= self.n_frames:
            raise IndexError(f"Frame index {frame_index} out of bounds (0-{self.n_frames-1})")
        
        return self.coordinates[frame_index]


class FreeEnergyMapParser:
    """
    Parser for free energy map data.
    """
    
    def __init__(self):
        """
        Initialize the free energy map parser.
        """
        self.grid = None
        self.values = None
        self.dimensions = None
        self.origin = None
        self.spacing = None
        
    def parse_file(self, file_path):
        """
        Parse a free energy map file and extract grid data.
        
        Parameters:
        -----------
        file_path : str
            Path to the free energy map file.
            
        Returns:
        --------
        dict
            Dictionary containing free energy map data.
        """
        # This is a placeholder for actual implementation
        # The actual format depends on the specific output format of meta-eABF simulations
        
        # For now, we'll assume a simple format with grid dimensions, origin, spacing, and values
        with open(file_path, 'r') as f:
            # Read header information
            line = f.readline().strip().split()
            self.dimensions = [int(x) for x in line]
            
            line = f.readline().strip().split()
            self.origin = [float(x) for x in line]
            
            line = f.readline().strip().split()
            self.spacing = [float(x) for x in line]
            
            # Read grid values
            total_points = np.prod(self.dimensions)
            self.values = np.zeros(total_points)
            
            i = 0
            for line in f:
                values = line.strip().split()
                for val in values:
                    if i < total_points:
                        self.values[i] = float(val)
                        i += 1
            
            # Reshape values to match grid dimensions
            self.values = self.values.reshape(self.dimensions)
            
        return {
            'dimensions': self.dimensions,
            'origin': self.origin,
            'spacing': self.spacing,
            'values': self.values
        }
    
    def get_value_at_point(self, point):
        """
        Get the free energy value at a specific point in the CV space.
        
        Parameters:
        -----------
        point : list or numpy.ndarray
            Coordinates in the CV space.
            
        Returns:
        --------
        float
            Free energy value at the specified point.
        """
        if self.values is None:
            raise ValueError("No free energy map loaded")
        
        # Convert point to grid indices
        indices = []
        for i, p in enumerate(point):
            idx = int((p - self.origin[i]) / self.spacing[i])
            if idx < 0 or idx >= self.dimensions[i]:
                # Point outside grid, return high energy
                return float('inf')
            indices.append(idx)
        
        # Return value at grid point
        return self.values[tuple(indices)]
    
    def interpolate_value_at_point(self, point):
        """
        Interpolate the free energy value at a specific point in the CV space.
        
        Parameters:
        -----------
        point : list or numpy.ndarray
            Coordinates in the CV space.
            
        Returns:
        --------
        float
            Interpolated free energy value at the specified point.
        """
        if self.values is None:
            raise ValueError("No free energy map loaded")
            
        # This is a simple implementation of trilinear interpolation for 3D
        # For other dimensions, a more general approach would be needed
        
        # Convert point to grid coordinates
        grid_coords = []
        for i, p in enumerate(point):
            coord = (p - self.origin[i]) / self.spacing[i]
            if coord < 0 or coord >= self.dimensions[i] - 1:
                # Point outside grid, return high energy
                return float('inf')
            grid_coords.append(coord)
        
        # Get the corner indices and weights for interpolation
        indices = []
        weights = []
        for coord in grid_coords:
            idx_low = int(coord)
            idx_high = idx_low + 1
            weight_high = coord - idx_low
            weight_low = 1.0 - weight_high
            indices.append((idx_low, idx_high))
            weights.append((weight_low, weight_high))
        
        # Perform trilinear interpolation
        result = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    idx = (indices[0][i], indices[1][j], indices[2][k])
                    weight = weights[0][i] * weights[1][j] * weights[2][k]
                    result += self.values[idx] * weight
                    
        return result