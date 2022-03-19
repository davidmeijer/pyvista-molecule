#!/usr/bin/env python3
import os
from argparse import ArgumentParser
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, GetMolFrags, AddHs, RemoveHs, AllChem
import pyvista as pv
import numpy as np

def cli():
    parser = ArgumentParser()
    parser.add_argument('-i', required=True, help='SMILES string or path to SDF file')
    return parser.parse_args().i

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

def rotate_x(degree, coords):
    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [1.0, np.cos(degree), -np.sin(degree)],
        [0.0, np.sin(degree), np.cos(degree)]
    ])
    return np.dot(rot_x, coords)

def rotate_y(degree, coords):
    rot_y = np.array([
        [np.cos(degree), 0.0, np.sin(degree)],
        [0.0, 1.0, 0.0],
        [-np.sin(degree), 0.0, np.cos(degree)]
    ])
    return np.dot(rot_y, coords)

def rotate_z(degree, coords):
    rot_z = np.array([
        [np.cos(degree), -np.sin(degree), 0.0],
        [np.sin(degree), np.cos(degree), 0.0],
        [0.0, 0.0, 1.0]
    ])
    return np.dot(rot_z, coords)

class Plotter:
    ATOM_COLORS = {
        'N': [0, 0, 1],
        'C': [100/256, 100/256, 100/256],
        'O': [1, 0, 0],
        'H': [1, 1, 1]
    }

    ATOM_SIZES = {
        'N': 0.5,
        'C': 0.5,
        'O': 0.5,
        'H': 0.3
    }

    def __init__(self, off_screen=True, transform=None):
        self.off_screen = off_screen
        self.plt = pv.Plotter(off_screen=off_screen)
        self.plt.background_color ='white'
        self.transform = (lambda x: x) if not transform else transform

    def plot_atoms(self, mol):
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            coords = self.transform(np.array([pos.x, pos.y, pos.z]))
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
            s_center = self.transform(np.array([s_pos.x, s_pos.y, s_pos.z]))
            e_center = self.transform(np.array([e_pos.x, e_pos.y, e_pos.z]))
            s_coords = s_center + self.ATOM_SIZES.get(s_symb, 0.3) * (e_center - s_center)
            e_coords = e_center + self.ATOM_SIZES.get(e_symb, 0.3) * (s_center - e_center)
            m_coords = centroid([s_coords, e_coords])

            tube_s = pv.Tube(
                pointa=s_center,
                pointb=m_coords,
                resolution=100,
                radius=radius,
                n_sides=100
            )
            self.plt.add_mesh(tube_s, color=self.ATOM_COLORS.get(s_symb, 'grey'))

            tube_e = pv.Tube(
                pointa=e_center,
                pointb=m_coords,
                resolution=100,
                radius=radius,
                n_sides=100
            )
            self.plt.add_mesh(tube_e, color=self.ATOM_COLORS.get(e_symb, 'grey'))

    def add_axes(self, length=5):
        x = pv.Line((-length, 0, 0), (length, 0, 0), resolution=1)
        self.plt.add_mesh(x, color='red')
        y = pv.Line((0, -length, 0), (0, length, 0))
        self.plt.add_mesh(y, color='green')
        z = pv.Line((0, 0, -length), (0, 0, length))
        self.plt.add_mesh(z, color='blue')
    
    def plot(self, name, transparent_background=True):
        if self.off_screen == False:
            self.plt.show()
        else:
            self.plt.screenshot(
                name + '.png',
                transparent_background=transparent_background,
                window_size=(5000, 5000)
            )

def main():
    input = cli()
    print(input)
    if not os.path.exists(input):
        mol = parse_smiles(input)
        mol = embed_conformer(mol)
    else:
        with open(input, 'r') as handle: sdf_string = handle.read()
        mol = Chem.MolFromMolBlock(sdf_string)

    # mol = RemoveHs(mol)
    plt = Plotter(off_screen=False)
    plt.plot_atoms(mol)
    plt.plot_bonds(mol)
    plt.plot('conformer', transparent_background=False)


if __name__ == '__main__':
    main()
