# File: src/data_loader.py
"""
Module for handling molecular data loading, primarily PDB structures,
to set up the initial configuration for the rMD simulation.
"""
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure

# Constants based on typical MD practices
# PDB files often require a slightly relaxed parser for non-standard formats
# The parser is instantiated globally to be ready for use.
PDB_PARSER = PDBParser(QUIET=True)


def load_pdb_structure(file_path: str, structure_id: str = "structure") -> Structure:
    """
    Loads a molecular structure from a Protein Data Bank (PDB) file.

    This function uses Bio.PDB to parse the PDB file, extracting the hierarchy
    (model, chain, residue, atom) into a Structure object.

    Args:
        file_path (str): The absolute or relative path to the PDB file.
        structure_id (str, optional): An identifier/name for the loaded structure.
                                      Defaults to "structure".

    Returns:
        Bio.PDB.Structure.Structure: A Structure object representing the molecular
                                      geometry and composition.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        IOError: If there is an issue parsing the PDB file content.
    """
    # Use Path for robust file path handling
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDB file not found at: {file_path}")

    try:
        structure = PDB_PARSER.get_structure(structure_id, path)
        return structure
    except Exception as e:
        # Catch any unexpected parsing errors from Bio.PDB
        raise IOError(f"Error parsing PDB file at {file_path}: {e}")

# End of src/data_loader.py