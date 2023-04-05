import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns

def dist_from_center(df, ball_center):
    '''Calculate distance from center of the ball
    
    For all coordinates, i.e. columns in df ending with '_x', '_y', '_z', 
    calculate the distance from `ball_center`.
    Returns dataframe with same indices as input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with raw data, must constain triplets of '_x', '_y', '_z' columns
    ball_center : np.array
        x, y, and z coordinate of the ball center

    Returns
    -------
    df_dist : pd.DataFrame
        Distances of each joint from the center.
    '''

    df_dist = pd.DataFrame(index=df.index)

    cols = [ c for c in df.columns if c[-2:] in [ '_x', '_y', '_z' ] ]

    for x, y, z in zip(cols[::3], cols[1::3], cols[2::3]):
        coords_aligned = df.loc[:, [x, y, z]].values - ball_center
        dist = np.linalg.norm(coords_aligned, axis=1)
        df_dist.loc[:, x[:-2]] = dist

    return df_dist