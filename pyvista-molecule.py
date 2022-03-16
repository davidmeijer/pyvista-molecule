#!/usr/bin/env python3
# https://docs.pyvista.org/
import time
from sys import argv
import pyvista as pv
from rdkit import Chem
import numpy as np


def main():
    t0 = time.time()

    path_to_sdf = argv[1]
    with open(path_to_sdf, 'r') as fo: sdf_string = fo.read()
    mol = Chem.MolFromMolBlock(sdf_string)

    atom_colors = {
        'N': [0, 0, 1], 
        'C': [100/256, 100/256, 100/256], 
        'O': [1, 0, 0]
    }
    
    plotter = pv.Plotter(off_screen=True)
    # plotter = pv.Plotter()

    nodes = []
    edges = []

    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        coords = [pos.x, pos.y, pos.z]
        sphere = pv.Sphere(
            radius=.3,
            center=coords,
            direction=coords,
            theta_resolution=90,
            phi_resolution=90
        )
        plotter.add_mesh(sphere, color=atom_colors.get(atom.GetSymbol(), 'grey'))
        nodes.append(coords)

    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([s, e])

        # s_atom, e_atom = mol.GetAtomWithIdx(s), mol.GetAtomWithIdx(e)
        # s_pos = mol.GetConformer().GetAtomPosition(s)
        # e_pos = mol.GetConformer().GetAtomPosition(e)
        # s_coords = np.array([s_pos.x, s_pos.y, s_pos.z])
        # e_coords = np.array([e_pos.x, e_pos.y, e_pos.z])
        # m_coords = (s_coords + e_coords) / 2

        # cylinder1 = pv.Cylinder(
        #     center=(s_coords + m_coords) / 2,
        #     direction=e_coords,
        #     radius=.1,
        #     height=np.sqrt(sum((s_coords - m_coords) ** 2)),
        #     resolution=100,
        #     capping=False
        # )
        # plotter.add_mesh(cylinder1, color=atom_colors.get(s_atom.GetSymbol(), 'grey'))

        # cylinder2 = pv.Cylinder(
        #     center=(e_coords + m_coords) / 2,
        #     direction=s_coords,
        #     radius=.1,
        #     height=np.sqrt(sum((e_coords - m_coords) ** 2)),
        #     resolution=100,
        #     capping=False
        # )
        # plotter.add_mesh(cylinder2, color=atom_colors.get(e_atom.GetSymbol(), 'grey'))

    edges = np.array(edges)
    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T
    mesh = pv.PolyData(nodes, edges_w_padding)
    plotter.add_mesh(
        mesh, 
        render_lines_as_tubes=True, 
        style='wireframe',
        line_width=5
    )

    plotter.screenshot('molecule.png', transparent_background=True, window_size=(5000, 5000))
    t1 = time.time()
    # plotter.show()

    print(f'total run time: {t1 - t0} s')
    print(0)

if __name__ == '__main__':
    main()