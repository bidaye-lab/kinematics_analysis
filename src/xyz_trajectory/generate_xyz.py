# -*- coding: utf-8 -*-
"""
@author: Nico Spiller
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def write_xyz(df, out_name, split, ball=None, scl=5.0):
    """Create xyz trajectory file(s) from CSV

    If split is set to 0, one file will be written.
    Otherwise, the output will be split into files with `split` frames each.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with x, y, z coordinates for each point
    out_name : path-like
        Name of the output file(s)
    split : int
        Number of frames to write to one file. Select 0 for no splitting
    ball : list, np.ndarray
        List of floats with x, y, z coordinates of the ball
    s : float, optional
        Scaling factor for all coordinates to get reasonable bond lengths, by default 5.0
    """

    # ensure that ball is a list
    if isinstance(ball, np.ndarray):
        ball = ball.tolist()

    # ensure that out_name is a Path object
    out_name = Path(out_name)

    print("INFO scaling distances by {}".format(scl))
    df = df * scl

    col_x = [i for i in df.columns if i.endswith("_x")]
    points = [i.rstrip("x").rstrip("_") for i in col_x]
    n = len(points)  # number of "atoms"
    
    if ball:
        print("INFO adding ball coordinates")
        n += 1
    print("INFO found {} points".format(n))

    print("INFO processing data ...")
    lines = []
    for i in df.index:
        lines.append("{}\n".format(n))  # first line: number of atoms
        lines.append("Frame {}\n".format(i))  # second line: commend

        for j in points:  # data lines: atom_name x_coord y_coord z_coord
            x = df.loc[i, j + "_x"]
            y = df.loc[i, j + "_y"]
            z = df.loc[i, j + "_z"]

            l = "{} {} {} {}\n".format(j, x, y, z)
            lines.append(l)

        if ball:
            x, y, z = [float(i) * scl for i in ball]
            l = "{} {} {} {}\n".format("Ball", x, y, z)
            lines.append(l)

    if split:  # write files with fixed number of frames
        xyz = lambda fid: out_name.with_name(out_name.stem + "_{}.xyz".format(fid))
        len_blk = split * (n + 2)  # each frame is n + 2 lines long
        fid = 0
        out = open(xyz(fid), "w")  # dummy file, will remain empty

        for i, l in enumerate(lines):
            if not i % len_blk:
                out.close()  # close previous file
                fid += 1
                print("INFO writing file {}".format(xyz(fid)))
                out = open(xyz(fid), "w")

            out.write(l)

        out.close()

        xyz(0).unlink()  # remove empty dummy file

    else:  # if 0, write one file
        xyz = out_name.with_suffix(".xyz")
        print("INFO writing file {}".format(xyz))
        with open(xyz, "w") as f:
            f.writelines(lines)

def csv2xyz(csv, split, ball):
    '''Wrapper for reading CSV and writing xyz trajectory file(s)

    Parameters
    ----------
    csv : path-like
        Path to CSV file
    split : int
        Number of frames to write to one file. Select 0 for no splitting
    ball : list
        List of floats with x, y, z coordinates of the ball
    '''

    print("INFO reading file {}".format(csv))
    df = pd.read_csv(csv)  # read CSV into pandas dataframe
    
    out_name = csv.with_suffix(".xyz")
    write_xyz(df, out_name, split, ball)

def run():
    # command line parser
    parser = argparse.ArgumentParser(
        description="""Create xyz trajectory file(s) from CSV"""
    )
    parser.add_argument("csv", help="Name of the CSV file")
    parser.add_argument(
        "-s",
        "--split",
        metavar="S",
        help="Split output files to S frames per file. Select 0 for no splitting. Default: 1400",
        default=1400,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--ball",
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Center position of the ball",
    )
    args = parser.parse_args()

    csv = Path(args.csv)  # input CSV
    split = args.split  # number of frames for splitting
    ball = args.ball  # center position of the ball

    csv2xyz(csv, split, ball)


if __name__ == "__main__":
    run()
