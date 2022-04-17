#!/usr/bin/env python3
"""
Author:         David Meijer
Description:    Draw gif of molecule rotating around x-axis.
Usage:          python3 draw_gif.py -i <input> -o <output_dir>
"""
import os
import argparse
from copy import deepcopy

import numpy as np
from rdkit.Chem import rdMolTransforms

import src


def cli() -> argparse.Namespace:
    """
    Command line interface for script.
    
    Returns
    -------
    argparse.Namespace: parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    # Required command line arguments.
    parser.add_argument("-i", "--input", required=True, type=str,
        help="SMILES string or path to SDF file"
    )
    parser.add_argument("-o", "--output", required=True, type=str,
        help="path to output directory"
    )
    # Optional command line arguments.
    parser.add_argument("-f", "--frames", required=False, type=int, 
        help="number of frames", default=30
    )
    parser.add_argument("-n", "--name", required=False, type=str,
        help="name of output file", default="conformer.png"
    )
    parser.add_argument("-wh", "--window-height", required=False, type=int,
        help="height of window", default=src.DEFAULT_WINDOW_SIZE[0]
    )
    parser.add_argument("-ww", "--window-width", required=False, type=int,
        help="width of window", default=src.DEFAULT_WINDOW_SIZE[1]
    )
    parser.add_argument("-sdr", "--slow_down_rate", required=False, type=int,
        help="slow down rate for frame change", default=1
    )
    parser.add_argument("-aa", "--add-axes", required=False,
        help="flag for adding axes", action="store_true"
    )
    return parser.parse_args()


def main() -> None:
    """
    Driver code.    
    """
    args = cli()

    if not os.path.exists(args.input):
        mol = src.parse_smiles(args.input)
    else:
        with open(args.input, "r") as handle: 
            sdf_string = handle.read()
        mol = src.parse_molblock(sdf_string)

    frames = []
    for _ in range(args.frames):
        mol = deepcopy(mol)
        rdMolTransforms.TransformConformer(
            mol.GetConformer(0), 
            src.rotation_matrix_x(2*np.pi/args.frames)
        )
        frames.append(mol)
    
    src.draw_gif(
        mols=frames, 
        dirpath=args.output, 
        window_size=(args.window_width, args.window_height),
        slow_down_rate=args.slow_down_rate,
        add_axes=args.add_axes
    )

if __name__ == '__main__':
    main()
