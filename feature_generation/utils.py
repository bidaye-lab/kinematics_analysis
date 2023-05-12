import pandas as pd
import numpy as np
from scipy.optimize import minimize

import matplotlib.pylab as plt
import seaborn as sns

import yaml
from pathlib import Path

################
## data handling

def load_config(config):
    '''Load config yml as dict.

    Parameters
    ----------
    config : str
        Path to config yml file

    Returns
    -------
    cfg : dict
        dictionary 
    '''

    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)


    return cfg



def load_data_hdf(cfg):
    '''Load data from location defined in cfg['datafile']

    Parameters
    ----------
    cfg : dict
        config dict with paths to files

    Returns
    -------
    data : dict
        Dictionary with genotypes as keys and pd.DataFrames with coordinates as values
    '''

    path = Path(cfg['datafile'])
    print('INFO: loading file {}'.format(path))
    dfs = pd.read_hdf(path)

    data = dict()
    for k, d in dfs.groupby('Genotype'):
        print('INFO: found genotype {}'.format(k))

        df = d.loc[:, 'flydata'].item()
        df = unify_columns(df)
        data[k] = df
    
    return data

def write_data_dict(data, path):
    '''Store dict of DataFrames as single parquet file in folder.

    Parameters
    ----------
    data : dict
        Keys are genotypes, values are dataframes
    path : path-like
        Path to store data
    '''

    l = []
    for gen, df in data.items():
        df.loc[:,'genotype'] = gen
        l.append(df)

    df_tot = pd.concat(l)

    print('INFO: writing file {}'.format(path))

    df_tot.to_parquet(path)

def load_data_dict(path):
    '''Load parquet file as dict of dataframes

    Parameters
    ----------

    path : path-like
        Path to parquet files
    
    Returns
    -------
    data : dict
        Dict containing dataframe per genotype
    '''

    print('INFO: loading file {}'.format(path))

    df_tot = pd.read_parquet(path)

    data = dict()
    # cycle through genotype
    for gen, df in df_tot.groupby('genotype'):
        data[gen] = df

    return data


def load_ball_centers(cfg):
    '''Load ball centers based on CSV files defined in config

    Parameters
    ----------
    cfg : dict
        config dict with paths to files


    Returns
    -------
    ball_centers : dict
        Mapping between genotype name and ball center (3d np.array)
    '''

    ball_centers = dict()
    for k, v in cfg.items():
        if k.endswith('_ball'):
            path = Path(v)
            df = pd.read_csv(path, index_col=0)
            xyz = df.iloc[0, :].values
            gen = k.split('_')[0]
            print('INFO: setting ball center for {} from file {}'.format(gen, path.name))
            print('      x = {:1.3f}, y = {:1.3f}, z = {:1.3f}'.format(*xyz))
            ball_centers[gen] = xyz

    return ball_centers

def unify_columns(df):
    '''Make leg names consistent in dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with coordinate data

    Returns
    -------
    df_uni : pd.DatFrame
        Data frame with unified column names
    '''

    print('INFO: Renaming stepcycle columns')
    leg2step = {
        'R1_stepcycle' : 'R-F_stepcycle',
        'R2_stepcycle' : 'R-M_stepcycle', 
        'R3_stepcycle' : 'R-H_stepcycle',
        'L1_stepcycle' : 'L-F_stepcycle',
        'L2_stepcycle' : 'L-M_stepcycle', 
        'L3_stepcycle' : 'L-H_stepcycle',
        'R1_stepcycle_ON' : 'R-F_stepcycle',
        'R2_stepcycle_ON' : 'R-M_stepcycle', 
        'R3_stepcycle_ON' : 'R-H_stepcycle',
        'L1_stepcycle_ON' : 'L-F_stepcycle',
        'L2_stepcycle_ON' : 'L-M_stepcycle', 
        'L3_stepcycle_ON' : 'L-H_stepcycle',
        }

    df_uni = df.rename(columns=leg2step)

    return df_uni

def filter_frames(df, f_0=400, f_f=1000, f_trl=1400):
    '''Return a copy of the dataframe only including frames between `f_0` and `f_f`
    Filtering is done on a by-trial basis, frames below or equal `f_0 % f_trl` and above `f_f % f_trl` are removed.
    Default values return only stimulus frames.

    Parameters
    ----------
    df : pd.DataFrame
        Frame numbers need to be stored in the `fnum` column
    f_0 : int, optional
        Remove frame numbers equal or below `f_0`, by default 400
    f_f : int, optional
        Remove frame numbers higher than `f_f`, by default 1000

    Returns
    -------
    df_filt : pd.DataFrame
        Filtered dataframe
    '''

    df = df.copy()
    
    df_filt = df.loc[ ((df.loc[:, 'fnum'] % f_trl) >= f_0) & ((df.loc[:, 'fnum'] % f_trl) < f_f) ]

    return df_filt

###########
## fit ball

def get_xyz_mean(df, point):
    '''Calculate the mean xyz coordinate for a given point

    Parameters
    ----------
    df : pd.DataFrame
        Coordinate data frame, must contain the x, y, and z columns for `point`
    point : str
        Used to identify uniquely a subset of columns.
        E.g. 'TaG' will average over all columns containing 'TaG_x', 'TaG_y', and 'TaG_z'

    Returns
    -------
    xyz : np.array
        3x1 numpy array with mean x, y, and z coordinates
    '''

    # construct xyz str based on point
    point_x, point_y, point_z = [ '{}_{}'.format(point, i) for i in 'xyz' ]

    # select xyz columns
    cols_x = [ c for c in df.columns if point_x in c ]
    cols_y = [ c for c in df.columns if point_y in c ]
    cols_z = [ c for c in df.columns if point_z in c ]

    # calculate mean
    x = df.loc[ :, cols_x ].values.mean()
    y = df.loc[ :, cols_y ].values.mean()
    z = df.loc[ :, cols_z ].values.mean()
    xyz = np.array([x, y, z])

    return xyz

def get_ball0(df, d=4.5):
    '''Generate initial guess based on average postitions of TaG and Notum.
    Calculates vector connecting average Notum with average TaG positions
    and sets lengs of vector equal to `d`

    Parameters
    ----------
    df : pd.DataFrame
        Coordinate data containing TaG and Notum positions
    d : float, optional
        length of Notum-TaG vector, by default 4.5

    Returns
    -------
    ball0 : np.array
        xyz positions of inital guess
    '''


    # mean of TaG and Notum posititons
    tag = get_xyz_mean(df, 'TaG')
    notum =  get_xyz_mean(df, 'Notum')

    # vector connecting means of Notum-TaG
    notum_tag = norm_vec(tag - notum)

    # initial guess 
    ball0 = notum + notum_tag * d

    return ball0


def cost_fun(x, l_pnts, l_perc):
    '''Calculate cost for distance of points from surface of sphere
    For a given list of `pnts` only select points in the percentile
    interval given in `l_perc`, then calculate the square of the distance
    from surface of sphere with center x=x[0], y=x[1], z=x[2] and radius x[3]

    Parameters
    ----------
    x : np.array
        4x1 array: x, y, z (center of sphere), and r (radius)
    l_pnts : list of np.arrays
        Each element is a Nx3 np array with xyz points
    l_perc : list of tuples
        List of percentile ranges pnts, same length as `l_pnts`
        e.g. (25, 75) selects points with radius between 25 and 75 percentile

    Returns
    -------
    cost : float
        Cost calculated as sum of squared distances from sphere surface
    '''

    # split input in ball center and ball radius
    ballc = x[:3]
    ballr = x[3]
    
    cost = 0
    for pnts, perc in zip(l_pnts, l_perc):
        # distance of all points from ball center
        r = np.linalg.norm(pnts - ballc, axis=1)

        # select points based on percentile
        a, b = np.nanpercentile(r, perc)
        r = r[ ( r > a ) & ( r < b )]

        # calculate cost (least squares)
        cost += np.sum((r - ballr)**2)

    return cost



def fit_ball(df, xyz0, r0, d_perc):
    '''Fit sphere based on TaG coordinates, the initial guess for the ball
    coordinates and percentiles for each leg indicating the points used for fitting

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with TaG xyz coordinates to be used for fitting
    xyz0 : np.array 3x1
        initial guess for ball center
    r0 : float
        initial guess for ball radius
    d_perc : dict
        Dict of tuples, maps leg names to percentile used for each leg
        e.g. 'R-F': (25, 75)

    Returns
    -------
    ball : np.array 3x1
        fitted xyz coordinates of ball center
    r : float
        fitted radius of ball
    '''
    
    # TaG points
    cols_x = [ c for c in df.columns if 'TaG_x' in c ]
    l_pnts, l_perc = [], []
    for c_x in cols_x:
        cs = [ c_x[:-1] + i for i in 'xyz' ]
        l_pnts.append(df.loc[:, cs].values)
        l_perc.append(d_perc[c_x[:3]])

    # initial guess
    x0 = np.array([*xyz0, r0])

    # optimize cost function
    res = minimize(cost_fun, x0, args=(l_pnts, l_perc), method='Nelder-Mead')
    ball, r = res.x[:3], res.x[3]

    return ball, r

def add_distance(df, ball):
    '''Add columns of distances from ball center for each xyz column triplet

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with xyz coordinates
    ball : np.array 3x1
        xyz coordinates of the ball

    Returns
    -------
    df : pd.DataFrame
        Data frame with distance columns added
    '''

    df = df.copy()

    # select all columns ending with x, y or z

    # cycle through all columns ending with `_x``
    cols_x = [c for c in df.columns if c[-2:] == '_x' ]
    for c_x in cols_x:

        # corresponding `_y` and `_z` columns
        c_y, c_z = '{}y'.format(c_x[:-1]), '{}z'.format(c_x[:-1])

        # calculate distance
        coords = df.loc[:, [c_x, c_y, c_z]].values
        dist = np.linalg.norm(coords - ball, axis=1)

        # write to df 
        r = '{}_r'.format(c_x[:-2])
        df.loc[:, r] = dist
    
    return df

#############
## Stepcycles
def get_r_median(df, d_perc):
    '''Calculate "median" of TaG positions for each leg
    The "median" is the mean of the values within the percentile interval 
    defined in `d_perc`

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing TaG_r columns
    d_perc : dict
        Percentile range in which to calculate the mean for each leg.
        e.g. 'R-M': (25, 75)

    Returns
    -------
    d_med : dict
        Mapping from leg to "Median", e.g. 'R-M': 2.98
    '''

    d_med = dict()

    # cycle through legs
    cols =  [ c for c in df.columns if 'TaG_r' in c ]
    for c in cols:

        perc = d_perc[c[:3]]

        # get "median" of r for given leg
        r = df.loc[:, c]
        a, b = np.nanpercentile(r, perc)
        r_m = r[(r>a) & (r<b)].mean()

        # fill dict
        leg = '-'.join(c.split('-')[:2])
        d_med[leg] = r_m

    return d_med

def add_stepcycle_pred(df, r_med, delta_r, min_on, min_off):
    '''Add columns with stepcycle predictions based on the TaG_r columns
    and multiple thresholds as explained below

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to which to add columns. Must contain TaG_r columns
    r_med : dict
        Mapping between leg and surface distance for that leg, e.g. 'R-M': 2.98
    delta_r : float
        distance above r_med to be considered on the ball
    min_on : int
        ignore on steps if frames less than min_on
    min_off : int
        ignore off steps if frames less than min_off

    Returns
    -------
    df : pd.DataFrame
        Data frames with stepcylce columns added
    '''

    df = df.copy()

    # cycle through legs
    cols =  [ c for c in df.columns if 'TaG_r' in c ]
    for c in cols:
        
        leg = '-'.join(c.split('-')[:2])

        # distances from center of ball
        r = df.loc[:, c]
        
        # on frames based on distance criterium
        on = r < (r_med[leg] + delta_r)

        # require min length of on and off series
        on_split = np.split(on, np.flatnonzero(np.diff(on))+1)
        for s in on_split:
            if s.sum() and (len(s) <= min_on):
                on.loc[s.index] = False
            elif not s.sum() and (len(s) <= min_off):
                on.loc[s.index] = True

        # add column to df
        df.loc[:, '{}_stepcycle'.format(leg)] = on
    
    return df

######################
## Coordinate handling

# TODO: remove this
# def add_distance(data, ball_centers):
#     '''Calculate distance from center of the ball
    
#     For all coordinates, i.e. columns in df ending with '_x', '_y', '_z', 
#     calculate the distance from `ball_center`.
#     Adds column ending with '_r' for each xyz triplet.
#     Assumes that '_x', '_y', '_z' columns always appear in this order.

#     Parameters
#     ----------
#     data : dict
#         Dict with dataframes with raw data, must constain triplets of '_x', '_y', '_z' columns
#     ball_centers : dict
#         dict with x, y, and z coordinate of the ball center, same keys as data
#     '''

#     for k in ball_centers.keys():
#         print('INFO: Adding distance columns for {}'.format(k))
#         df = data[k]
#         c = ball_centers[k]

#         cols = [ c for c in df.columns if c[-2:] in [ '_x', '_y', '_z' ] ]

#         for x, y, z in zip(cols[::3], cols[1::3], cols[2::3]):
#             coords_aligned = df.loc[:, [x, y, z]].values - c
#             dist = np.linalg.norm(coords_aligned, axis=1)
#             r = '{}_r'.format(x[:-2])
#             df.loc[:, r] = dist

# data = utl.load_data_hdf(cfg)
# ball_centers = utl.load_ball_centers(cfg)

# ball_centers['P9LT'] = ball
# utl.add_distance(data, ball_centers)

# # tranform coordinates
# df = data['P9LT'].groupby('flynum').get_group(1).groupby('tnum').get_group(1)
# df = utl.filter_frames(df)


def norm_vec(x):
    '''Return normalized vector

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    x_norm : np.ndarray
        Array with same direction but length 1
    '''
      
    x_norm = x / np.linalg.norm(x)

    return x_norm


def fit_plane_thc(df):
    '''Fit a plane through the set of ThC points in a given DataFrame
    Returns `a`, `b`, and `c` for the fitted equation `a*x + b*y + c = z`

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ending with 'ThC_x', 'ThC_y', and 'ThC_z'

    Returns
    -------
    a, b, c : floats
        fitted parameters
    '''

    # collect columns ending with ThC_x
    cols_x = [c for c in df.columns if c[-5:] == 'ThC_x' ]

    # fill x, y, z coordinate arrays
    xs, ys, zs = [], [], []
    for c_x in cols_x:
        c_y, c_z = '{}y'.format(c_x[:-1]), '{}z'.format(c_x[:-1])
        xs.extend(df.loc[:, c_x])    
        ys.extend(df.loc[:, c_y])
        zs.extend(df.loc[:, c_z])

    # fit linear equations for a*x + b*y + c = z
    A = np.vstack([xs, ys, np.ones(len(xs), )]).T
    b = np.array(zs)
    res = np.linalg.lstsq(A, b, rcond=None)

    # print results
    a, b, c = res[0]
    print('INFO: Fitted plane through ThC points')
    print('      a = {:1.2f}, b = {:1.2f}, c = {:1.2f}, residual = {:1.1f}'.format(a, b, c, res[1].item()))

    return a, b, c


def construct_basis(df, a, b, c):
    '''Construct basis based on plane through ThC points, Notum, and L/R-WH

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with Notum, R-WH, and L-WH coordinates
    a : float
        Parameter for the plane defined by ThC plane
    a : float
        Parameter for the plane defined by ThC plane
    a : float
        Parameter for the plane defined by ThC plane

    Returns
    -------
    T : 3x3 np.array
        Tranformation matrix from old to new coordinate system
    center : 3x1 np.array
        center of new coordinate system
    '''

    # intermediate basis (necessary to project points on ThC plane)
    n = norm_vec(np.array([-a, -b, 1]))
    o = np.array([0, 0, c])
    e1 = norm_vec(np.array([1, 0, a]))
    e2 = np.cross(e1, n)

    # mean of points used for final basis
    cols = [ 'Notum_{}'.format(i) for i in 'xyz' ]
    notum = df.loc[:, cols].mean().values
    notum_std = df.loc[:, cols].std().values
    cols = [ 'R-WH_{}'.format(i) for i in 'xyz' ]
    rwh = df.loc[:, cols].mean().values
    rwh_std = df.loc[:, cols].std().values
    cols = [ 'L-WH_{}'.format(i) for i in 'xyz' ]
    lwh = df.loc[:, cols].mean().values
    lwh_std = df.loc[:, cols].std().values

    # print info
    print('INFO: Constructing basis based on')
    print('      ThC plane a = {: .3f} | b = {: .3f} | c = {: .3f}'.format(a, b, c))
    print('      notum x = {: .3f} ({:.3f}) | y = {: .3f} ({:.3f}) | z = {: .3f} ({:.3f})'.format(
        notum[0], notum_std[0], notum[1], notum_std[1], notum[2], notum_std[2]))
    print('      R-WH  x = {: .3f} ({:.3f}) | y = {: .3f} ({:.3f}) | z = {: .3f} ({:.3f})'.format(
        rwh[0], rwh_std[0], rwh[1], rwh_std[1], rwh[2], rwh_std[2]))   
    print('      L-WH  x = {: .3f} ({:.3f}) | y = {: .3f} ({:.3f}) | z = {: .3f} ({:.3f})'.format(
        lwh[0], lwh_std[0], lwh[1], lwh_std[1], lwh[2], lwh_std[2]))
    # projection of points in to intermediate basis
    proj12 = lambda p: o + np.dot(p - o, e1) * e1 + np.dot(p - o, e2) * e2

    # z axis normal to the mean ThC points, positive towards notum
    ez = n if np.dot(n, notum - o) > 0 else -n 
    # x axis along WH (R-WH) positive
    ex = norm_vec(proj12(rwh) - proj12(lwh))
    # y axis perpendicular to WH connection, positive towards head
    ey = np.cross(ez, ex)

    # construct transformation matrix
    T = np.vstack([ex, ey, ez]).T

    # center is mean of WH
    center = np.mean([proj12(rwh), proj12(lwh)], axis=0) 

    return T, center


def transform_to_flycentric(df):
    '''Transform all columns ending with `_x`, `_y`, or `_z` to fly-centric coordinates.
    See `fit_plane_thc` and `construct_basis` for details.
    Returns copy with transformed coordinates

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with `_x`, `_y`, or `_z` columns in arbitrary coordinates

    Returns
    -------
    df_trans : pd.DataFrame
        Transformed data frame
    '''

    df_trans = df.copy()
    
    # get tranformation matrix and new center
    a, b, c = fit_plane_thc(df)
    T, center = construct_basis(df, a, b, c)

    print('INFO: Transforming to fly-centric coordinates')

    # all columns ending with `_x``
    cols_x = [c for c in df.columns if c[-2:] == '_x' ]

    for c_x in cols_x:

        # corresponding `_y` and `_z` columns
        c_y, c_z = '{}y'.format(c_x[:-1]), '{}z'.format(c_x[:-1])

        # basis tranformation
        xyz = df.loc[:, [c_x, c_y, c_z]].values
        xyz -= center
        xyz = xyz @ T

        # overwrite data
        df_trans.loc[df.index, [c_x, c_y, c_z]] = xyz
    
    return df_trans


#####################
## Plotting functions

def save(fig, path):

    if path:
        fig.savefig(path)
        plt.close(fig)

def plot_coord_system(df, joints=['WH', 'ThC', 'Notum'], return_lims=False, lims=(), marker='o', marker_size=5, swing_gray=False, path=''):
    '''Plot distribution of joints projected on planes defined by basis vectors

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing xyz coordinates for the points defined in `joints`
    joints : list, optional
        Joints to plot, by default ['WH', 'ThC', 'Notum']
    return_lims : bool, optional
        If true, do not show plot but only return axis limits (useful for movie creation), by default False
    lims : tuple, optional
        Tuple of axis limits, designed to be used with `return_lims` from previous plot, by default ()
    marker : str, optional
        Matplotlib marker to be used, by default 'o'
    marker_size : int, optional
        Matplotlib marker size, by default 5
    swing_gray : bool, optional
        If true, plot swing phase in gray. Only works if `joints` are part of a leg, by default False
    path : str, optional
        Filepath to save figure, if set plot is not showsn, by default ''

    Returns
    -------
    lims : tuple, optional
        Only returned when `return_lims` is True, otherwise return None
    '''

    fig, axarr = plt.subplots(ncols=3, figsize=(15, 5))

    cols = [ c for c in df.columns if c[-1] in 'xyz' ]
    cols = [ c for c in cols if c.split('_')[0].split('-')[-1] in joints ]
    
    xlabel = 'x (left/right)'
    ylabel = 'y (posterior/anterior)'
    zlabel = 'z (ventral/dorsal)'

    for col_xyz in zip(cols[::3], cols[1::3], cols[2::3]):

        xyz = df.loc[:, col_xyz].values

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        leg = col_xyz[0][:3]
        if swing_gray:
            s = df.loc[:, '{}_stepcycle'.format(leg)]
        else:
            s = np.ones(len(df)).astype(bool)

        ax = axarr[0]
        ax.scatter(y[s], z[s], marker=marker, s=marker_size, label=col_xyz[0][:-2])
        ax.scatter(y[~s], z[~s], marker=marker, s=marker_size, c='gray')
        ax.set_xlabel(ylabel)
        ax.set_ylabel(zlabel)
        ax.set_title('side view')
        ax.legend(loc='upper right')

        ax = axarr[1]
        ax.scatter(x[s], y[s], marker=marker, s=marker_size)
        ax.scatter(x[~s], y[~s], marker=marker, s=marker_size, c='gray')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('top view')

        ax = axarr[2]
        ax.scatter(x[s], z[s], marker=marker, s=marker_size)
        ax.scatter(x[~s], z[~s], marker=marker, s=marker_size, c='gray')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(zlabel)
        ax.set_title('front view')
    
    for ax in axarr:
        ax.axvline(0, ls=':', c='gray')
        ax.axhline(0, ls=':', c='gray')
    
    if lims:
        for ax, x, y in zip(axarr, lims[0], lims[1]):
            ax.set_xlim(x)
            ax.set_ylim(y)
            
    if return_lims:
        xlims = [ ax.get_xlim() for ax in axarr ]
        ylims = [ ax.get_ylim() for ax in axarr ]
        plt.close(fig)
        return xlims, ylims
    
    fig.tight_layout()

    save(fig, path)

def plot_r_distr(df, col_match, d_perc={}, xlims=(None, None), path=''):

    # construct xyz str based on joint
    cols = [ c for c in df.columns if col_match in c ]
    
    fig, axmat = plt.subplots(nrows=len(cols)//2, ncols=2, figsize=(10, len(cols)*1.5))

    for ax, c in zip(axmat.T.flatten(), cols):

        r = df.loc[:, c].values
        perc = d_perc.get(c[:3], [5, 95])

        # create arrays for three intervals
        a, b = np.nanpercentile(r, perc)
        r1 = r[r<a]
        r2 = r[(r>a) & (r<b)]
        r3 = r[r>b]

        # plot
        sns.histplot([r1, r2, r3], ax=ax, legend=False)
        ax.set_title(c)
        ax.set_xlim(xlims)

    fig.tight_layout()
    save(fig, path)

def plot_stepcycle_pred(df, d_med, delta_r, path=''):

    cols =  [ c for c in df.columns if 'TaG_r' in c ]
    fig, axarr = plt.subplots(nrows=len(cols), figsize=(20, 20))

    for ax, c in zip(axarr, cols,):

        leg = '-'.join(c.split('-')[:2])

        # on/off ball predictions
        on = df.loc[:, '{}_stepcycle'.format(leg)]
        off = ~on

        # distance from ball center
        r = df.loc[:, c]

        sns.scatterplot(r.loc[on], ax=ax)
        sns.scatterplot(r.loc[off], ax=ax)
        
        # plot "median" and thresh
        r_m = d_med[leg]
        ax.axhline(r_m, c='gray')
        ax.axhline(r_m + delta_r, c='gray', ls='--')
        
    save(fig, path)


def plot_stepcycle_pred_grid(df, d_med, delta_r, path=''):

    cols =  [ c for c in df.columns if 'TaG_r' in c ]
    trials = df.loc[:, 'tnum'].unique()
    fig, axmat = plt.subplots(nrows=len(trials), ncols=len(cols), figsize=(40, 4*len(trials)))

    for axarr, c in zip(axmat.T, cols,):
        for ax, (t, df_t) in zip(axarr, df.groupby('tnum')):

            leg = '-'.join(c.split('-')[:2])

            # on/off ball predictions
            on = df_t.loc[:, '{}_stepcycle'.format(leg)]
            off = ~on

            # distance from ball center
            r = df_t.loc[:, c]

            sns.scatterplot(r.loc[on], ax=ax)
            sns.scatterplot(r.loc[off], ax=ax)
            sns.lineplot(r, ax=ax, color='gray', lw=0.5)
            
            # plot "median" and thresh
            r_m = d_med[leg]
            ax.axhline(r_m, c='gray')
            ax.axhline(r_m + delta_r, c='gray', ls='--')

            ax.set_title('leg {} | trial {}'.format(leg, t))
    
    fig.tight_layout()
        
    save(fig, path)