import numpy as np
from scipy.optimize import minimize


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

def cost_fun_ball(x, l_pnts, l_perc):
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

def fit_ball(df, d_perc, s_ball0=4.7, s_r0=3.5):
    '''Fit sphere based on TaG coordinates, the initial guess for the ball
    coordinates and percentiles for each leg indicating the points used for fitting

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with TaG xyz coordinates to be used for fitting
    d_perc : dict
        Dict of tuples, maps leg names to percentile used for each leg
        e.g. 'R-F': (25, 75)
    s_ball0 : float, optinonal
        scaling factor: ball0 is s_ball0 * distance WH along Notum->avg TaG positions
    s_r0 : float, optional
        scaling factor: r0 is s_r0 * distance WH

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


    # wing hinge distance
    lwh = get_xyz_mean(df, 'L-WH')
    rwh = get_xyz_mean(df, 'R-WH')
    dwh = np.linalg.norm(lwh - rwh)

    # get initial guess for ball0 based on average notum/tag and WH distance
    xyz0 = get_ball0(df, d=s_ball0*dwh)
    # get r0 based on WH distance
    r0 = s_r0 * dwh
    
    print('INFO: scaling WH distance = {:1.3f} with factors s_ball0 = {:1.3f} and s_r0 = {:1.3f}'.format(dwh, s_ball0, s_r0))
    print('INFO: initial guess for ball center x = {:1.3f} y = {:1.3f} z = {:1.3f} | radius {:1.3f}'.format(*xyz0, r0))

    # initial guess
    x0 = np.array([*xyz0, r0])

    # optimize cost function
    res = minimize(cost_fun_ball, x0, args=(l_pnts, l_perc), method='Nelder-Mead')
    ball, r = res.x[:3], res.x[3]

    return ball, r