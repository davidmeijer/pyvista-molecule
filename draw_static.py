#!/usr/bin/env python3
"""
Author:         David Meijer
Description:    Draw 3D conformer of molecule.
Usage:          python3 draw_static.py -i <input> -o <output_dir>
"""
import os
import argparse

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
    parser.add_argument("-n", "--name", required=False, type=str,
        help="name of output file", default="conformer.png"
    )
    parser.add_argument("-wh", "--window-height", required=False, type=int,
        help="height of window", default=src.DEFAULT_WINDOW_SIZE[0]
    )
    parser.add_argument("-ww", "--window-width", required=False, type=int,
        help="width of window", default=src.DEFAULT_WINDOW_SIZE[1]
    )
    parser.add_argument("-tb", "--transparent-background", required=False,
        help="flag for setting transparent background", action="store_true"
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

    src.draw(
        mol, 
        fpath=os.path.join(args.output, args.name), 
        window_size=(args.window_width, args.window_height),
        transparent_background=args.transparent_background,
        add_axes=args.add_axes
    )


if __name__ == '__main__':
    main()