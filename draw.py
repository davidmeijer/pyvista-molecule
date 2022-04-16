#!/usr/bin/env python3
import os
from argparse import ArgumentParser
from copy import deepcopy

from matplotlib import offsetbox
from rdkit import Chem
from rdkit.Chem import (
    MolFromSmiles, 
    GetMolFrags, 
    AddHs, 
    RemoveHs, 
    AllChem, 
    rdMolTransforms
)
import pyvista as pv
import numpy as np
import imageio
from tqdm import tqdm



def cli():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, 
        help="SMILES string or path to SDF file"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="path to output directory"
    )
    return parser.parse_args()

def parse_smiles(smiles):
    mol = MolFromSmiles(smiles)
    if mol == None: raise ValueError('mol is None')
    if '.' in smiles:
        try: fragments = list(GetMolFrags(mol, asMols=True))
        except Exception: raise ValueError('mol fragments are None')
        else: 
            fragments.sort(key=lambda fragment: fragment.GetNumHeavyAtoms())
            mol = fragments[-1]
    return mol

def embed_conformer(mol):
    mol = AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(mol) # MMFF94
    try: _ = mol.GetConformer()
    except: raise ValueError('conformer is None')
    return mol

def atom_dist(a, b):
    return np.sqrt(sum((a - b) ** 2))

def centroid(arr):
    return np.sum(arr, axis=0) / len(arr)

def rotate_x(degree):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(degree), -np.sin(degree), 0.0],
        [0.0, np.sin(degree), np.cos(degree), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotate_y(degree):
    return np.array([
        [np.cos(degree), 0.0, np.sin(degree), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(degree), 0.0, np.cos(degree), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotate_z(degree):
    return np.array([
        [np.cos(degree), -np.sin(degree), 0.0, 0.0],
        [np.sin(degree), np.cos(degree), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

class Plotter:
    ATOM_COLORS = {
        'N': [ 71/256,  89/256, 230/256],
        'C': [211/256, 211/256, 211/256],
        'O': [228/256,  31/256,  36/256],
        'H': [1, 1, 1]
    }

    ATOM_SIZES = {
        'N': 0.5,
        'C': 0.5,
        'O': 0.5,
        'H': 0.3
    }

    def __init__(self, off_screen=True):
        self.off_screen = off_screen
        self.plt = pv.Plotter(off_screen=off_screen)
        self.plt.background_color = "white"

    def plot_atoms(self, mol):
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            coords = np.array([pos.x, pos.y, pos.z])
            symb = atom.GetSymbol()

            sphere = pv.Sphere(
                radius=self.ATOM_SIZES.get(symb, 0.3),
                center=coords,
                direction=coords,
                theta_resolution=90,
                phi_resolution=90
            )
            self.plt.add_mesh(sphere, color=self.ATOM_COLORS.get(symb, 'grey'))

    def plot_bonds(self, mol, radius=0.3):
        for bond in mol.GetBonds():
            s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            s_atom, e_atom = mol.GetAtomWithIdx(s), mol.GetAtomWithIdx(e)
            s_symb, e_symb = s_atom.GetSymbol(), e_atom.GetSymbol()
            s_pos = mol.GetConformer().GetAtomPosition(s)
            e_pos = mol.GetConformer().GetAtomPosition(e)
            s_center = np.array([s_pos.x, s_pos.y, s_pos.z])
            e_center = np.array([e_pos.x, e_pos.y, e_pos.z])
            s_coords = s_center + self.ATOM_SIZES.get(s_symb, 0.3) * (e_center - s_center)
            e_coords = e_center + self.ATOM_SIZES.get(e_symb, 0.3) * (s_center - e_center)
            m_coords = centroid([s_coords, e_coords])

            tube_s = pv.Tube(
                pointa=s_center,
                pointb=m_coords,
                resolution=100,
                n_sides=100,
                radius=radius
            )
            self.plt.add_mesh(tube_s, color=self.ATOM_COLORS.get(s_symb, 'grey'))

            tube_e = pv.Tube(
                pointa=e_center,
                pointb=m_coords,
                resolution=100,
                n_sides=100,
                radius=radius
            )
            self.plt.add_mesh(tube_e, color=self.ATOM_COLORS.get(e_symb, 'grey'))

    def add_axes(self, length=5, opacity=1.0):
        x = pv.Line((-length, 0, 0), (length, 0, 0), resolution=1)
        self.plt.add_mesh(x, color='red', opacity=opacity)
        y = pv.Line((0, -length, 0), (0, length, 0))
        self.plt.add_mesh(y, color='green', opacity=opacity)
        z = pv.Line((0, 0, -length), (0, 0, length))
        self.plt.add_mesh(z, color='blue', opacity=opacity)

    def frame(self, mol):
        """Add axes to make sure perspective is static."""
        axes_length = np.max(np.abs(mol.GetConformer(0).GetPositions()))
        self.add_axes(length=axes_length + 5, opacity=0.0)
    
    def plot(
        self, 
        fpath, 
        transparent_background=True,
        window_size=[1024, 768],
        add_axes=False
    ):
        if add_axes:
            self.add_axes()
        if self.off_screen == False:
            self.plt.show()
        else:
            self.plt.screenshot(
                fpath,
                transparent_background=transparent_background,
                window_size=window_size
            )

def draw(mol, fpath, window_size, add_axes=False):
    plt = Plotter(off_screen=True)
    plt.plot_atoms(mol)
    plt.plot_bonds(mol)
    plt.plot(
        fpath, 
        transparent_background=True, 
        window_size=window_size,
        add_axes=add_axes
    )

def draw_gif(
    mols, 
    dirpath, 
    window_size, 
    slow_down_rate=1,
    add_axes=False
):
    fpath = os.path.join(dirpath, "mol.gif")
    with imageio.get_writer(fpath, mode='I') as writer:
        for i, step in tqdm(enumerate(mols)):
            plt = Plotter(off_screen=True)
            plt.frame(step)
            plt.plot_atoms(step)
            plt.plot_bonds(step)
            frame = os.path.join(dirpath, f"frame_{i}.png")
            plt.plot(
                frame, 
                transparent_background=True,
                window_size=window_size,
                add_axes=add_axes
            )
            for _ in range(slow_down_rate):
                image = imageio.imread(frame)
                writer.append_data(image)
            os.remove(frame)

def main():
    args = cli()
    input = args.input
    print(input)
    if not os.path.exists(input):
        mol = parse_smiles(input)
        mol = embed_conformer(mol)
        mol = RemoveHs(mol)
    else:
        with open(input, 'r') as handle: sdf_string = handle.read()
        mol = Chem.MolFromMolBlock(sdf_string)

    # draw(mol, fpath="conformer.png", window_size=[5000, 5000])

    steps, mols = 30, []
    for _ in range(steps + 1):
        mol = deepcopy(mol)
        rdMolTransforms.TransformConformer(
            mol.GetConformer(0), 
            rotate_x(2*np.pi/steps)
        )
        mols.append(mol)
    
    draw_gif(
        mols, 
        dirpath=args.output, 
        window_size=[250, 250],
        slow_down_rate=1,
        add_axes=True
    )

if __name__ == '__main__':
    main()
