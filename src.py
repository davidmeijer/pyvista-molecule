import os
import typing as ty
from unittest.mock import DEFAULT

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import pyvista as pv
import numpy as np
import imageio


# =============================================================================
# Default values.
# =============================================================================
DEFAULT_ATOM_SIZE = 0.3
DEFAULT_ATOM_COLOR = (0.0, 0.0, 0.0)
DEFAULT_BOND_RADIUS = 0.3
DEFAULT_WINDOW_SIZE = (800, 800)
BLACK = (0.0, 0.0, 0.0)
WHITE = (1.0, 1.0, 1.0)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


# =============================================================================
# Type definitions.
# =============================================================================
Color = ty.Tuple[float, float, float]
Size = float


# =============================================================================
# RDKit utilities.
# =============================================================================
def smiles_to_mol(smiles: str) -> Chem.Mol:
    """
    Parse SMILES string into a RDKit molecule. 

    Arguments
    ---------
    smiles (str): SMILES string.

    Returns
    -------
    mol (rdkit.Chem.Mol): RDKit molecule.

    Returns only biggest fragment if SMILES is fragmented (i.e., includes '.').
    """
    mol = Chem.MolFromSmiles(smiles)
    # Raise ValueError if mol was not properly parsed by RDKit.
    if mol == None: 
        raise ValueError(
            f"{smiles} was not properly parsed into a RDKit molecule!"
        )
    if '.' in smiles:
        try: 
            fragments = list(Chem.GetMolFrags(mol, asMols=True))
        except Exception: 
            raise ValueError("Could not parse fragments from RDKit molecule!")
        else: 
            # Retrieve biggest fragment as RDKit molecule from parsed SMILES.
            fragments.sort(key=lambda fragment: fragment.GetNumHeavyAtoms())
            mol = fragments[-1]
    return mol


def embed_conformer(mol: Chem.Mol) -> Chem.Mol:
    """
    Embed MMFF94 optimized 3D-conformer in RDKit molecule. 

    Arguments
    ---------
    mol (rdkit.Chem.Mol): RDKit molecule.

    Returns
    -------
    mol (rdkit.Chem.Mol): RDKit molecule with single embedded 3D-conformer.
    """
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(mol)  # MMFF94
    # Raise ValueError if 3D-conformer embedding failed.
    try: 
        _ = mol.GetConformer()
    except: 
        raise ValueError("Could not embed conformer in RDKit molecule!")
    return mol


def parse_smiles(smiles: str, remove_hs: bool = True) -> Chem.Mol:
    """
    Parse SMILES string to RDKit molecule with embedded conformer.
    
    Arguments
    ---------
    smiles (str): SMILES string.
    remove_hs (bool): Remove hydrogens from molecule.

    Returns
    -------
    mol (rdkit.Chem.Mol): RDKit molecule with embedded conformer.
    """
    mol = smiles_to_mol(smiles)
    mol = embed_conformer(mol)
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    return mol


def parse_molblock(molblock: str, remove_hs: bool = True) -> Chem.Mol:
    """
    Parse MolBlock string to RDKit molecule with embedded conformer.

    Arguments
    ---------
    molblock (str): MolBlock string.
    remove_hs (bool): Remove hydrogens from molecule.

    Returns
    -------
    mol (rdkit.Chem.Mol): RDKit molecule.
    """
    mol = Chem.MolFromMolBlock(molblock)
    # Raise ValueError if mol was not properly parsed by RDKit.
    if mol == None: 
        raise ValueError(
            f"{molblock} was not properly parsed into a RDKit molecule!"
        )
    mol = embed_conformer(mol)
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    return mol


# =============================================================================
# Geometry utilities.
# =============================================================================
def centroid(arr: np.array) -> np.array:
    """
    Calculates centroid of x, y, z coordinates of point cloud. 

    Arguments
    ---------
    arr (np.array): Array of x, y, z coordinates with dimensions (N, 3).

    Returns
    -------
    centroid (np.array): Centroid of point cloud with dimensions (1, 3). 
    """
    return np.sum(arr, axis=0) / len(arr)


def rotation_matrix_x(r: float) -> np.array:
    """
    Calculates rotation matrix for rotation around x-axis.

    Arguments
    ---------
    r (float): Rotation angle in radians.

    Returns
    -------
    rot_matrix (np.array): Rotation matrix for rotation around x-axis.

    For more information on 4x4 rotation matrices:
    http://www.cs.cmu.edu/afs/cs/academic/class/15462-s10/www/lec-slides/lec04.pdf
    """
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(r), -np.sin(r), 0.0],
        [0.0, np.sin(r), np.cos(r), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rotation_matrix_y(r: float) -> np.array:
    """
    Calculates rotation matrix for rotation around y-axis.
    
    Arguments
    ---------
    r (float): Rotation angle in radians.
    
    Returns
    -------
    rot_matrix (np.array): Rotation matrix for rotation around y-axis.

    For more information on 4x4 rotation matrices:
    http://www.cs.cmu.edu/afs/cs/academic/class/15462-s10/www/lec-slides/lec04.pdf
    """
    return np.array([
        [np.cos(r), 0.0, np.sin(r), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(r), 0.0, np.cos(r), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rotation_matrix_z(r: float) -> np.array:
    """
    Calculates rotation matrix for rotation around z-axis.
    
    Arguments
    ---------
    r (float): Rotation angle in radians.
    
    Returns
    -------
    rot_matrix (np.array): Rotation matrix for rotation around z-axis.

    For more information on 4x4 rotation matrices:
    http://www.cs.cmu.edu/afs/cs/academic/class/15462-s10/www/lec-slides/lec04.pdf
    """
    return np.array([
        [np.cos(r), -np.sin(r), 0.0, 0.0],
        [np.sin(r), np.cos(r), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


# =============================================================================
# Visualization.
# =============================================================================
class Plotter:
    """
    Class for plotting 3D depictions of molecules.
    """
    # Default atom colors if no custom colors are provided. Bond color will
    # default to color of closest neighboring atom.
    DEFAULT_ATOM_COLORS = {
        "N": [71/256, 89/256, 230/256],
        "C": [211/256, 211/256, 211/256],
        "O": [228/256, 31/256, 36/256],
        "H": WHITE
    }
    # Default atom sizes if no custom sizes are provided.
    DEFAULT_ATOM_SIZES = {
        "N": 0.5,
        "C": 0.5,
        "O": 0.5,
        "H": 0.3
    }

    def __init__(
        self, 
        background_color: Color = (1.0, 1.0, 1.0),
        off_screen: bool = True
    ) -> None:
        """
        Initialize Plotter.
        
        Arguments
        ---------
        background_color (Color): Background color of plot (default: white).
        off_screen (bool): Whether to plot to off-screen buffer (default: True).
        """
        self.off_screen = off_screen
        self.plt = pv.Plotter(off_screen=off_screen)
        self.plt.background_color = background_color

    def get_atom_size(self, atom: str) -> float:
        """
        Get size of atom.

        Arguments
        ---------
        atom (str): Atom symbol.

        Returns
        -------
        size (float): Size of atom.
        """
        if atom in self.DEFAULT_ATOM_SIZES:
            return self.DEFAULT_ATOM_SIZES[atom]
        else:
            print(f"Warning: Atom {atom} not found in default atom sizes.")
            return DEFAULT_ATOM_SIZE

    def get_atom_color(self, atom: str) -> Color:
        """
        Get color of atom.

        Arguments
        ---------
        atom (str): Atom symbol.

        Returns
        -------
        color (Color): Color of atom.
        """
        if atom in self.DEFAULT_ATOM_COLORS:
            return self.DEFAULT_ATOM_COLORS[atom]
        else:
            print(f"Warning: Atom {atom} not found in default atom colors.")
            return DEFAULT_ATOM_COLOR

    def plot_atoms(
        self, 
        mol: Chem.Mol, 
        colors: ty.Optional[ty.Dict[int, Color]] = None,
        sizes: ty.Optional[ty.Dict[int, Size]] = None,
        theta_resolution: int = 90,
        phi_resolution: int = 90
    ) -> None:
        """
        Plot atoms of RDKit molecule.
        
        Arguments
        ---------
        mol (rdkit.Chem.Mol): RDKit molecule.
        colors (ty.Dict[int, Color]): Dictionary of atom indices and colors
            (default: None).
        sizes (ty.Dict[int, Size]): Dictionary of atom indices and sizes
            (default: None).
        theta_resolution (int): Resolution of theta angle (default: 90).
        phi_resolution (int): Resolution of phi angle (default: 90).

        Note: RDKit molecule atom indices are used as reference.
        """
        for atom in mol.GetAtoms():
            atom_index = atom.GetIdx()

            pos = mol.GetConformer().GetAtomPosition(atom_index)
            coords = np.array([pos.x, pos.y, pos.z])
            symb = atom.GetSymbol()

            # Set atom color.
            if colors is not None:
                color = colors.get(atom_index, self.get_atom_color(symb))
            else:
                color = self.get_atom_color(symb)

            # Set atom size.
            if sizes is not None:
                size = sizes.get(atom_index, self.get_atom_size(symb))
            else:
                size = self.get_atom_size(symb)

            # Draw atom.
            sphere = pv.Sphere(
                radius=size,
                center=coords,
                direction=coords,
                theta_resolution=theta_resolution,
                phi_resolution=phi_resolution
            )
            self.plt.add_mesh(sphere, color=color)

    def plot_bonds(
        self, 
        mol: Chem.Mol, 
        colors: ty.Optional[ty.Dict[int, Color]] = None,
        radii: ty.Optional[ty.Dict[int, Size]] = None,
        resolution: int = 100,
        n_sides: int = 100
    ) -> None:
        """
        Plot bonds of RDKit molecule.

        Arguments
        ---------
        mol (rdkit.Chem.Mol): RDKit molecule.
        colors (ty.Dict[int, Color]): Dictionary of bond indices and colors
            (default: None).
        radii (ty.Dict[int, Size]): Dictionary of bond indices and radii
            (default: None).
        resolution (int): Resolution of bond cylinders (default: 100).
        n_sides (int): Number of sides of bond cylinders (default: 100).
        """
        for bond in mol.GetBonds():
            bond_index = bond.GetIdx()

            s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            s_atom, e_atom = mol.GetAtomWithIdx(s), mol.GetAtomWithIdx(e)
            s_symb, e_symb = s_atom.GetSymbol(), e_atom.GetSymbol()
            s_pos = mol.GetConformer().GetAtomPosition(s)
            e_pos = mol.GetConformer().GetAtomPosition(e)
            s_center = np.array([s_pos.x, s_pos.y, s_pos.z])
            e_center = np.array([e_pos.x, e_pos.y, e_pos.z])
            s_coords = s_center + self.get_atom_size(s_symb) * (e_center - s_center)
            e_coords = e_center + self.get_atom_size(e_symb) * (s_center - e_center)
            m_coords = centroid([s_coords, e_coords])

            # Set bond color.
            if colors is not None:
                if bond_index in colors:
                    s_color = e_color = colors[bond_index]
                else:
                    s_color = self.get_atom_color(s_symb)
                    e_color = self.get_atom_color(e_symb)
            else:
                s_color = self.get_atom_color(s_symb)
                e_color = self.get_atom_color(e_symb)

            # Set bond radius.
            if radii is not None:
                s_radius = e_radius = radii.get(bond_index, DEFAULT_BOND_RADIUS)
            else:
                s_radius = e_radius = DEFAULT_BOND_RADIUS

            # Draw bond.
            tube_s = pv.Tube(
                pointa=s_center,
                pointb=m_coords,
                resolution=resolution,
                n_sides=n_sides,
                radius=s_radius
            )
            self.plt.add_mesh(tube_s, color=s_color)

            tube_e = pv.Tube(
                pointa=e_center,
                pointb=m_coords,
                resolution=resolution,
                n_sides=n_sides,
                radius=e_radius
            )
            self.plt.add_mesh(tube_e, color=e_color)

    def add_axes(
        self, 
        mol: Chem.Mol,
        opacity_x_axis: float = 1.0,
        opacity_y_axis: float = 1.0,
        opacity_z_axis: float = 1.0,
        color_x_axis: Color = RED,
        color_y_axis: Color = GREEN,
        color_z_axis: Color = BLUE,
        axis_length: ty.Optional[float] = None
    ) -> None:
        """
        Add xyz aces to plot.

        Arguments
        ---------
        mol (rdkit.Chem.Mol): RDKit molecule.
        opacity_x_axis (float): Opacity of x axis (default: 1.0).
        opacity_y_axis (float): Opacity of y axis (default: 1.0).
        opacity_z_axis (float): Opacity of z axis (default: 1.0).
        color_x_axis (Color): Color of x axis (default: red).
        color_y_axis (Color): Color of y axis (default: green). 
        color_z_axis (Color): Color of z axis (default: blue).
        """
        # Get maximum wingspan of molecule and add buffer of 5, if not provided.
        if axis_length is None:
            axis_length = np.max(np.abs(mol.GetConformer(0).GetPositions())) + 5
        # Add red x-axis.
        x = pv.Line((-axis_length, 0, 0), (axis_length, 0, 0), resolution=1)
        self.plt.add_mesh(x, color=color_x_axis, opacity=opacity_x_axis)
        # Add green y-axis.
        y = pv.Line((0, -axis_length, 0), (0, axis_length, 0))
        self.plt.add_mesh(y, color=color_y_axis, opacity=opacity_y_axis)
        # Add blue z-axis.
        z = pv.Line((0, 0, -axis_length), (0, 0, axis_length))
        self.plt.add_mesh(z, color=color_z_axis, opacity=opacity_z_axis)
    
    def draw(
        self, 
        fpath: ty.Optional[str] = None, 
        transparent_background: bool = False,
        window_size: ty.Tuple[int, int] = DEFAULT_WINDOW_SIZE,
    ) -> None:
        """
        Draw plot and save to file or show plotter.

        Arguments
        ---------
        fpath (str): Path to save plot to (default: None).
        transparent_background (bool): Whether to make background transparent
            (default: False).
        window_size (ty.Tuple[int, int]): Size of window (default: (800, 800)).
        """
        if self.off_screen == False:
            self.plt.show()
        else:
            if fpath is None:
                raise ValueError("Must specify fpath to save plot to.")
            self.plt.screenshot(
                fpath,
                transparent_background=transparent_background,
                window_size=window_size
            )


def draw(
    mol: Chem.Mol,
    fpath: ty.Optional[str] = None, 
    window_size: ty.Tuple[int, int] = DEFAULT_WINDOW_SIZE,
    transparent_background: bool = False,
    atom_colors: ty.Optional[ty.Dict[int, Color]] = None,
    atom_sizes: ty.Optional[ty.Dict[int, Size]] = None,
    atom_theta_resolution: int = 90,
    atom_phi_resolution: int = 90,
    bond_colors: ty.Optional[ty.Dict[int, Color]] = None,
    bond_radii: ty.Optional[ty.Dict[int, Size]] = None,
    bond_resolution: int = 100,
    bond_n_sides: int = 100,
    add_axes: bool = False,
    opacity_x_axis: float = 1.0,
    opacity_y_axis: float = 1.0,
    opacity_z_axis: float = 1.0,
    color_x_axis: Color = RED,
    color_y_axis: Color = GREEN,
    color_z_axis: Color = BLUE
) -> None:
    """
    Draw molecule.
    
    Arguments
    ---------
    mol (rdkit.Chem.Mol): RDKit molecule.
    fpath (str): Path to save plot to (default: None).
    window_size (ty.Tuple[int, int]): Size of window (default: (800, 800)).
    transparent_background (bool): Whether to make background transparent
    atom_colors (ty.Dict[int, Color]): Atom colors (default: None).
    atom_sizes (ty.Dict[int, Size]): Atom sizes (default: None).
    atom_theta_resolution (int): Theta resolution of atoms (default: 90).
    atom_phi_resolution (int): Phi resolution of atoms (default: 90).
    bond_colors (ty.Dict[int, Color]): Bond colors (default: None).
    bond_radii (ty.Dict[int, Size]): Bond radii (default: None).
    bond_resolution (int): Resolution of bonds (default: 100).
    bond_n_sides (int): Number of sides of bonds (default: 100).
    add_axes (bool): Whether to add axes to plot (default: False).
    opacity_x_axis (float): Opacity of x axis (default: 1.0).
    opacity_y_axis (float): Opacity of y axis (default: 1.0).
    opacity_z_axis (float): Opacity of z axis (default: 1.0).
    color_x_axis (Color): Color of x axis (default: red).
    color_y_axis (Color): Color of y axis (default: green).
    color_z_axis (Color): Color of z axis (default: blue).
    """
    plt = Plotter(off_screen=True)
    plt.plot_atoms(
        mol,
        colors=atom_colors,
        sizes=atom_sizes,
        theta_resolution=atom_theta_resolution,
        phi_resolution=atom_phi_resolution
    )
    plt.plot_bonds(
        mol,
        colors=bond_colors,
        radii=bond_radii,
        resolution=bond_resolution,
        n_sides=bond_n_sides
    )
    if add_axes:
        plt.add_axes(
            mol,
            opacity_x_axis=opacity_x_axis,
            opacity_y_axis=opacity_y_axis,
            opacity_z_axis=opacity_z_axis,
            color_x_axis=color_x_axis,
            color_y_axis=color_y_axis,
            color_z_axis=color_z_axis
        )
    plt.draw(
        fpath, 
        transparent_background=transparent_background, 
        window_size=window_size
    )


def draw_gif(
    mols: ty.List[Chem.Mol], 
    dirpath: str, 
    window_size: ty.Tuple[int, int] = DEFAULT_WINDOW_SIZE,
    slow_down_rate: int = 1,
    atom_colors: ty.Optional[ty.List[ty.Optional[ty.Dict[int, Color]]]] = None,
    atom_sizes: ty.Optional[ty.List[ty.Optional[ty.Dict[int, Size]]]] = None,
    atom_theta_resolution: ty.Union[int, ty.List[int]] = 90,
    atom_phi_resolution: ty.Union[int, ty.List[int]] = 90,
    bond_colors: ty.Optional[ty.List[ty.Optional[ty.Dict[int, Color]]]] = None,
    bond_radii: ty.Optional[ty.List[ty.Optional[ty.Dict[int, Size]]]] = None,
    bond_resolution: ty.Union[int, ty.List[int]] = 100,
    bond_n_sides: ty.Union[int, ty.List[int]] = 100,
    add_axes: bool = False,
    opacity_x_axis: float = 1.0,
    opacity_y_axis: float = 1.0,
    opacity_z_axis: float = 1.0,
    color_x_axis: Color = RED,
    color_y_axis: Color = GREEN,
    color_z_axis: Color = BLUE
) -> None:
    """
    Draw molecule movie as gif.
    
    Arguments
    ---------
    mols (ty.List[rdkit.Chem.Mol]): RDKit molecules that act as frames.
    dirpath (str): Directory to save gif to.
    window_size (ty.Tuple[int, int]): Size of window (default: (800, 800)).
    slow_down_rate (int): Slow down rate of gif (default: 1).
    atom_colors (ty.List[ty.Dict[int, Color]]): Atom colors per frame 
        (default: None).
    atom_sizes (ty.List[ty.Dict[int, Size]]): Atom sizes per frame 
        (default: None).
    atom_theta_resolution (ty.Union[int, ty.List[int]]): Theta resolution of
        atoms per frame, if list, or for all frames, if int (default: 90).
    atom_phi_resolution (ty.Union[int, ty.List[int]]): Phi resolution of 
        atoms per frame, if list, or for all frames, if int (default: 90).
    bond_colors (ty.List[ty.Dict[int, Color]]): Bond colors per frame
        (default: None).
    bond_radii (ty.List[ty.Dict[int, Size]]): Bond radii per frame
        (default: None).
    bond_resolution (ty.Union[int, ty.List[int]]): Resolution of bonds per
        frame, if list, or for all frames, if int (default: 100).
    bond_n_sides (ty.Union[int, ty.List[int]]): Number of sides of bonds per
        frame, if list, or for all frames, if int (default: 100).
    add_axes (bool): Whether to add axes to plot (default: False).
    opacity_x_axis (float): Opacity of x axis (default: 1.0).
    opacity_y_axis (float): Opacity of y axis (default: 1.0).
    opacity_z_axis (float): Opacity of z axis (default: 1.0).
    color_x_axis (Color): Color of x axis (default: red).
    color_y_axis (Color): Color of y axis (default: green).
    color_z_axis (Color): Color of z axis (default: blue).
    """
    axis_length = 5.0

    # Create list of parameter if parameter is static for all frames.
    if isinstance(atom_theta_resolution, int):
        atom_theta_resolution = [atom_theta_resolution] * len(mols)
    if isinstance(atom_phi_resolution, int):
        atom_phi_resolution = [atom_phi_resolution] * len(mols)
    if isinstance(bond_resolution, int):
        bond_resolution = [bond_resolution] * len(mols)
    if isinstance(bond_n_sides, int):
        bond_n_sides = [bond_n_sides] * len(mols)

    fpath = os.path.join(dirpath, "mol.gif")
    with imageio.get_writer(fpath, mode='I') as writer:
        for i, step in tqdm(enumerate(mols)):
            plt = Plotter(off_screen=True)
            # Add transparent axes to make sure axes don't shift between frames.
            plt.add_axes(
                mol=step,
                opacity_x_axis=0.0,
                opacity_y_axis=0.0,
                opacity_z_axis=0.0,
                axis_length=axis_length,
            )
            plt.plot_atoms(
                mol=step,
                colors=atom_colors[i] if atom_colors is not None else None,
                sizes=atom_sizes[i] if atom_sizes is not None else None,
                theta_resolution=atom_theta_resolution[i],
                phi_resolution=atom_phi_resolution[i]
            )
            plt.plot_bonds(
                mol=step,
                colors=bond_colors[i] if bond_colors is not None else None,
                radii=bond_radii[i] if bond_radii is not None else None,
                resolution=bond_resolution[i],
                n_sides=bond_n_sides[i]
            )
            if add_axes:
                plt.add_axes(
                    mol=step,
                    opacity_x_axis=opacity_x_axis,
                    opacity_y_axis=opacity_y_axis,
                    opacity_z_axis=opacity_z_axis,
                    color_x_axis=color_x_axis,
                    color_y_axis=color_y_axis,
                    color_z_axis=color_z_axis
                )
            # Save frame to file.
            frame = os.path.join(dirpath, f"frame_{i}.png")
            plt.draw(
                frame, 
                transparent_background=False,
                window_size=window_size
            )
            # Add frame to gif. 
            for _ in range(slow_down_rate):
                image = imageio.imread(frame)
                writer.append_data(image)
            # Remove saved frame file.
            os.remove(frame)
