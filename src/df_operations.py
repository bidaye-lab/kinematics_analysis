import numpy as np

def unify_columns(df):
    """Make leg names consistent in dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with coordinate data

    Returns
    -------
    df_uni : pd.DatFrame
        Data frame with unified column names
    """

    print("INFO: Renaming stepcycle columns")
    leg2step = {
        "R1_stepcycle": "R-F_stepcycle",
        "R2_stepcycle": "R-M_stepcycle",
        "R3_stepcycle": "R-H_stepcycle",
        "L1_stepcycle": "L-F_stepcycle",
        "L2_stepcycle": "L-M_stepcycle",
        "L3_stepcycle": "L-H_stepcycle",
        "R1_stepcycle_ON": "R-F_stepcycle",
        "R2_stepcycle_ON": "R-M_stepcycle",
        "R3_stepcycle_ON": "R-H_stepcycle",
        "L1_stepcycle_ON": "L-F_stepcycle",
        "L2_stepcycle_ON": "L-M_stepcycle",
        "L3_stepcycle_ON": "L-H_stepcycle",
    }

    df_uni = df.rename(columns=leg2step)

    return df_uni


def filter_frames(df, f_0=400, f_f=1000, f_trl=1400):
    """Return a copy of the dataframe only including frames between `f_0` and `f_f`
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
    f_trl : int, optional
        Number of frames per trial, by default 1400

    Returns
    -------
    df_filt : pd.DataFrame
        Filtered dataframe
    """

    df = df.copy()

    df_filt = df.loc[
        ((df.loc[:, "fnum"] % f_trl) >= f_0) & ((df.loc[:, "fnum"] % f_trl) < f_f)
    ]

    return df_filt


def remove_stepcycle_predictions(df):
    """Remove stepcycle predictions from dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with coordinate data

    Returns
    -------
    df : pd.DataFrame
        Data frame without stepcycle predictions
    """

    # remove previous stepcycle predictions
    cols = [c for c in df.columns if c.endswith("stepcycle")]

    df = df.drop(columns=cols)

    return df

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

def add_stepcycle_pred(df, r_med, d_delta_r, min_on, min_off):
    '''Add columns with stepcycle predictions based on the TaG_r columns
    and multiple thresholds as explained below

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to which to add columns. Must contain TaG_r columns
    r_med : dict
        Mapping between leg and surface distance for that leg, e.g. 'R-M': 2.98
    d_delta_r : dict
        Cutoff distance above r_med to be considered on the ball.
        Dict to define on per leg basis, e.g. d_delta_r['R-M'] = 0.05
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
    for leg, delta_r in d_delta_r.items():
        
        col = f'{leg}-TaG_r'

        # distances from center of ball
        r = df.loc[:, col]
        
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