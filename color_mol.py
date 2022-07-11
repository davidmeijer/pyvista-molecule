#!/usr/bin/env python3
from sys import argv 
import json

import pyvista as pv
import numpy as np

from rdkit import Chem
from pikachu.general import read_smiles
from pikachu.drawing import drawing
from pikachu.drawing.colours import RANDOM_PALETTE_1

import src


def main() -> None:
    """
    Driver code.
    """
    input = argv[1]
    with open(input, "r") as handle: input = json.load(handle) 
    
    mol = read_smiles(input["substrate"]["mol"])
    drawer = drawing.draw_multiple(mol, coords_only=True)

    src.draw(
        drawer,
        atom_sizes={a.nr: 3.0 for a in drawer.structure.graph},
        bond_radii={bond_nr: 3.0 for bond_nr, _ in drawer.structure.bonds.items()},
        fpath="./out/erythromycin_1.png",
        window_size=(3200, 3200),
        transparent_background=True
    )

    atom_colors = {}
    for i, sub in enumerate(input["monomers"]):
        color = RANDOM_PALETTE_1[i % len(RANDOM_PALETTE_1)]
        for atom_index in sub["map"]:
            atom_colors[atom_index] = color

    src.draw(
        drawer,
        atom_sizes={a.nr: 3.0 for a in drawer.structure.graph},
        atom_colors=atom_colors,
        bond_radii={bond_nr: 3.0 for bond_nr, _ in drawer.structure.bonds.items()},
        fpath="./out/erythromycin_2.png",
        window_size=(3200, 3200),
        transparent_background=True
    )

    print(RANDOM_PALETTE_1)

    monomers = []
    for i, sub in enumerate(input["monomers"]):
        color = RANDOM_PALETTE_1[i % len(RANDOM_PALETTE_1)]
        monomers.append((sub, color))

    pks = []
    other = []
    chain = None
    for sub, color in monomers:
        if "polyketide_sequence" in sub["other"]:
            pks.append((sub, color))
            chain = sub["other"]["polyketide_sequence"]
        else:
            other.append((sub, color))

    colors = [
       "#ffe119",
        "#4363d8",
        "#f58231",
        '#911eb4',
        "#46f0f0",
        "#f032e6",
    ][::-1]

    plt = src.Plotter()
    for i, color in enumerate(colors):
        print(color)
        coords = np.array([1.0 + i, 0.0, 0.0])
        sphere = pv.Sphere(
            radius=0.5,
            center=coords,
            direction=coords,
            theta_resolution=90,
            phi_resolution=90
        )
        plt.plt.add_mesh(sphere, color=color)
    for i, (sub, color) in enumerate(other):
        coords = np.array([len(pks) - 1 + i + 0.2, 0.0, 0.0])
        sphere = pv.Sphere(
            radius=0.4,
            center=coords,
            direction=coords,
            theta_resolution=90,
            phi_resolution=90
        )
        plt.plt.add_mesh(sphere, color=color)
    # plt.plt.camera.position = (0.0, 0.0, -20)
    tube_s = pv.Tube(
        pointa=np.array([1.0, 0.0, 0.0]),
        pointb=np.array([9.0, 0.0, 0.0]),
        resolution=100,
        n_sides=100,
        radius=0.15
    )
    plt.plt.add_mesh(tube_s, color=[211/256, 211/256, 211/256])
    plt.plt.camera_position = 'xy'
    plt.draw("./out/erythromycin_3.png", transparent_background=True, window_size=(3200, 3200))



    plt = src.Plotter()
    tube_s = pv.Tube(
        pointa=np.array([1.0, 0.0, 0.0]),
        pointb=np.array([1.0, 0.0, 3.0]),
        resolution=100,
        n_sides=300,
        radius=4
    )
    plt.plt.add_mesh(tube_s, color=[211/256, 211/256, 211/256])
    tube_s = pv.Tube(
        pointa=np.array([1.0, 0.0, 3.1]),
        pointb=np.array([1.0, 0.0, 6.0]),
        resolution=100,
        n_sides=300,
        radius=4
    )
    plt.plt.add_mesh(tube_s, color=[211/256, 211/256, 211/256])
    tube_s = pv.Tube(
        pointa=np.array([1.0, 0.0, 6.1]),
        pointb=np.array([1.0, 0.0, 9.0]),
        resolution=100,
        n_sides=300,
        radius=4
    )
    plt.plt.add_mesh(tube_s, color=[211/256, 211/256, 211/256])
    tube_s = pv.Tube(
        pointa=np.array([1.0, 0.0, 0.0]),
        pointb=np.array([1.0, 0.0, 9.0]),
        resolution=100,
        n_sides=300,
        radius=3.9
    )
    plt.plt.add_mesh(tube_s, color=[0, 0, 0])
    plt.plt.camera_position = 'xz'
    plt.draw("./out/erythromycin_4.png", transparent_background=True, window_size=(3200, 3200))



if __name__ == "__main__":
    main()
