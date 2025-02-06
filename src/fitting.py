import numpy as np
import math
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import plotly.graph_objects as go
from scipy.signal import find_peaks
from tqdm import tqdm 
from moviepy.editor import ImageSequenceClip

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


def extract_leg_points(df, prefix):
    cols_x = [c for c in df.columns if f'{prefix}_x' in c]
    points = []
    for c_x in cols_x:
        cs = [c_x[:-1] + i for i in 'xyz']
        points.append(df.loc[:, cs].values)
    return points


def joint_angle_filtering(df):
    
    filtered_parts = []
    for i in range((len(df['tnum'].unique()))):
        start_idx = 400 + (i * 1400)
        end_idx = start_idx + 600
        filtered_parts.append(df.iloc[start_idx:end_idx])

    filtered_df = pd.concat(filtered_parts)

    
    indices_R3B,_ = find_peaks(filtered_df['R3B_flex'], height=0, prominence=25)
    indices_R2B,_ = find_peaks(filtered_df['R2B_flex'], height=0, prominence=25)
    indices_R1B,_ = find_peaks(filtered_df['R1B_flex'], height=0, prominence=25)

    indices_L3B,_ = find_peaks(filtered_df['L3B_flex'], height=0,prominence=25)
    indices_L2B,_ = find_peaks(filtered_df['L2B_flex'], height=0, prominence=25)
    indices_L1B,_ = find_peaks(filtered_df['L1B_flex'], height=0, prominence=25)

 
    L1B_df = filtered_df.iloc[indices_L1B]
    L2B_df = filtered_df.iloc[indices_L2B]
    L3B_df = filtered_df.iloc[indices_L3B]
    R1B_df = filtered_df.iloc[indices_R1B]
    R2B_df = filtered_df.iloc[indices_R2B]
    R3B_df = filtered_df.iloc[indices_R3B]
    
    return L1B_df, L2B_df, L3B_df, R1B_df, R2B_df, R3B_df

def tag_mean_leg(prefix):
    mean = (prefix[0][0]+prefix[0][1]+prefix[0][2]+prefix[0][3]+prefix[0][4]+prefix[0][5])/6
    return mean

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
    L1B_df, L2B_df, L3B_df, R1B_df, R2B_df, R3B_df = joint_angle_filtering(df)

    # Extract leg points for each leg segment
    L_F_pnts = extract_leg_points(L1B_df, 'L-F-TaG')
    L_M_pnts = extract_leg_points(L2B_df, 'L-M-TaG')
    L_H_pnts = extract_leg_points(L3B_df, 'L-H-TaG')
    R_F_pnts = extract_leg_points(R1B_df, 'R-F-TaG')
    R_M_pnts = extract_leg_points(R2B_df, 'R-M-TaG')
    R_H_pnts = extract_leg_points(R3B_df, 'R-H-TaG')

    # Find the smallest length among all points
    smallest = min(L_F_pnts[0].shape[0], L_M_pnts[0].shape[0], L_H_pnts[0].shape[0], 
                   R_F_pnts[0].shape[0], R_M_pnts[0].shape[0], R_H_pnts[0].shape[0])

    # Truncate all points to the smallest length
    L_F_pnts[0] = L_F_pnts[0][:smallest]
    L_M_pnts[0] = L_M_pnts[0][:smallest]
    L_H_pnts[0] = L_H_pnts[0][:smallest]
    R_F_pnts[0] = R_F_pnts[0][:smallest]
    R_M_pnts[0] = R_M_pnts[0][:smallest]
    R_H_pnts[0] = R_H_pnts[0][:smallest]

    tag = (tag_mean_leg(L_F_pnts) + tag_mean_leg(L_M_pnts) + tag_mean_leg(L_H_pnts) + tag_mean_leg(R_F_pnts) + tag_mean_leg(R_M_pnts) + tag_mean_leg(R_H_pnts)) / 6
    notum =  get_xyz_mean(df, 'Notum')

    # vector connecting means of Notum-TaG
    notum_tag = norm_vec(tag - notum)

    # initial guess 
    ball0 = notum + notum_tag * 4.5
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
    
    L1B_df, L2B_df, L3B_df, R1B_df, R2B_df, R3B_df = joint_angle_filtering(df)

    # Extract leg points for each leg segment
    L_F_pnts = extract_leg_points(L1B_df, 'L-F-TaG')
    L_M_pnts = extract_leg_points(L2B_df, 'L-M-TaG')
    L_H_pnts = extract_leg_points(L3B_df, 'L-H-TaG')
    R_F_pnts = extract_leg_points(R1B_df, 'R-F-TaG')
    R_M_pnts = extract_leg_points(R2B_df, 'R-M-TaG')
    R_H_pnts = extract_leg_points(R3B_df, 'R-H-TaG')

    # Find the smallest length among all points
    smallest = min(L_F_pnts[0].shape[0], L_M_pnts[0].shape[0], L_H_pnts[0].shape[0], 
                   R_F_pnts[0].shape[0], R_M_pnts[0].shape[0], R_H_pnts[0].shape[0])

    # Truncate all points to the smallest length
    L_F_pnts[0] = L_F_pnts[0][:smallest]
    L_M_pnts[0] = L_M_pnts[0][:smallest]
    L_H_pnts[0] = L_H_pnts[0][:smallest]
    R_F_pnts[0] = R_F_pnts[0][:smallest]
    R_M_pnts[0] = R_M_pnts[0][:smallest]
    R_H_pnts[0] = R_H_pnts[0][:smallest]

    l_pnts = [L_F_pnts[0], L_M_pnts[0], L_H_pnts[0], R_F_pnts[0], R_M_pnts[0], R_H_pnts[0]]
    
    cols_x = [c for c in df.columns if 'TaG_x' in c]
    l_perc = [d_perc[c_x[:3]] for c_x in cols_x]


    # wing hinge distance
    lwh = get_xyz_mean(df, 'L-WH')
    rwh = get_xyz_mean(df, 'R-WH')
    dwh = np.linalg.norm(lwh - rwh)

    # get initial guess for ball0 based on average notum/tag and WH distance
    xyz0 = get_ball0(df, d=s_ball0*dwh)
    # get r0 based on WH distance
    r0 = s_r0 * dwh

    # initial guess
    x0 = np.array([*xyz0, r0])

    # optimize cost function
    res = minimize(cost_fun_ball, x0, args=(l_pnts, l_perc), method='Nelder-Mead')
    ball, r = res.x[:3], res.x[3]

    return ball, r


def plot_walk_vid(df, path, flynum, genotype, trial, view, ball):
  
    frames_path = os.path.join(path, 'frames')
    os.makedirs(frames_path, exist_ok=True)

    notum = get_xyz_mean(df, 'Notum')
    lwh = get_xyz_mean(df, 'L-WH')
    rwh = get_xyz_mean(df, 'R-WH')


    frame_files = []
    
    start = list(df['fnum'])[600*(trial-1)]
    end = start + 600
    fnums = range(start,end)
    
    L_F_stances = []
    L_M_stances = []
    L_H_stances = []
    R_F_stances = []
    R_M_stances = []
    R_H_stances = []
    
    for frame in tqdm(fnums):
        L_F = get_xyz_mean(df[df['fnum'] == frame], 'L-F-TaG')
        L_M = get_xyz_mean(df[df['fnum'] == frame], 'L-M-TaG')
        L_H = get_xyz_mean(df[df['fnum'] == frame], 'L-H-TaG')
        R_F = get_xyz_mean(df[df['fnum'] == frame], 'R-F-TaG')
        R_M = get_xyz_mean(df[df['fnum'] == frame], 'R-M-TaG')
        R_H = get_xyz_mean(df[df['fnum'] == frame], 'R-H-TaG')
            
        if df[df['fnum'] == frame]['L-F_stepcycle'][frame] == True:
            L_F_stances.append(L_F)
        if df[df['fnum'] == frame]['L-M_stepcycle'][frame] == True:
            L_M_stances.append(L_M)
        if df[df['fnum'] == frame]['L-H_stepcycle'][frame] == True:
            L_H_stances.append(L_H)
        if df[df['fnum'] == frame]['R-F_stepcycle'][frame] == True:
            R_F_stances.append(R_F)
        if df[df['fnum'] == frame]['R-M_stepcycle'][frame] == True:
            R_M_stances.append(R_M)
        if df[df['fnum'] == frame]['R-H_stepcycle'][frame] == True:
            R_H_stances.append(R_H)
            
        L_F_TiTa = get_xyz_mean(df[df['fnum'] == frame], 'L-F-TiTa')
        L_M_TiTa = get_xyz_mean(df[df['fnum'] == frame], 'L-M-TiTa')
        L_H_TiTa = get_xyz_mean(df[df['fnum'] == frame], 'L-H-TiTa')
        R_F_TiTa = get_xyz_mean(df[df['fnum'] == frame], 'R-F-TiTa')
        R_M_TiTa = get_xyz_mean(df[df['fnum'] == frame], 'R-M-TiTa')
        R_H_TiTa = get_xyz_mean(df[df['fnum'] == frame], 'R-H-TiTa')
        
        L_F_FeTi = get_xyz_mean(df[df['fnum'] == frame], 'L-F-FeTi')
        L_M_FeTi = get_xyz_mean(df[df['fnum'] == frame], 'L-M-FeTi')
        L_H_FeTi = get_xyz_mean(df[df['fnum'] == frame], 'L-H-FeTi')
        R_F_FeTi = get_xyz_mean(df[df['fnum'] == frame], 'R-F-FeTi')
        R_M_FeTi = get_xyz_mean(df[df['fnum'] == frame], 'R-M-FeTi')
        R_H_FeTi = get_xyz_mean(df[df['fnum'] == frame], 'R-H-FeTi')
        
        L_F_CTr = get_xyz_mean(df[df['fnum'] == frame], 'L-F-CTr')
        L_M_CTr = get_xyz_mean(df[df['fnum'] == frame], 'L-M-CTr')
        L_H_CTr = get_xyz_mean(df[df['fnum'] == frame], 'L-H-CTr')
        R_F_CTr = get_xyz_mean(df[df['fnum'] == frame], 'R-F-CTr')
        R_M_CTr = get_xyz_mean(df[df['fnum'] == frame], 'R-M-CTr')
        R_H_CTr = get_xyz_mean(df[df['fnum'] == frame], 'R-H-CTr')

        L_F_ThC = get_xyz_mean(df[df['fnum'] == frame], 'L-F-ThC')
        L_M_ThC = get_xyz_mean(df[df['fnum'] == frame], 'L-M-ThC')
        L_H_ThC = get_xyz_mean(df[df['fnum'] == frame], 'L-H-ThC')
        R_F_ThC = get_xyz_mean(df[df['fnum'] == frame], 'R-F-ThC')
        R_M_ThC = get_xyz_mean(df[df['fnum'] == frame], 'R-M-ThC')
        R_H_ThC = get_xyz_mean(df[df['fnum'] == frame], 'R-H-ThC')

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        size = 3

        ax.scatter(*L_F, color='green', label='L-F-TaG', s=size)
        ax.scatter(*L_M, color='blue', label='L-M-TaG', s=size)
        ax.scatter(*L_H, color='red', label='L-H-TaG', s=size)
        
        ax.scatter(*R_F, color='yellow', label='R-F-TaG', s=size)
        ax.scatter(*R_M, color='purple', label='R-M-TaG', s=size)
        ax.scatter(*R_H, color='pink', label='R-H-TaG', s=size)
        
        ax.scatter(*L_F_TiTa, color='green', label='L-F-TiTa', s=size)
        ax.scatter(*L_M_TiTa, color='blue', label='L-M-TiTa', s=size)
        ax.scatter(*L_H_TiTa, color='red', label='L-H-TiTa', s=size)
        
        ax.scatter(*R_F_TiTa, color='yellow', label='R-F-TiTa', s=size)
        ax.scatter(*R_M_TiTa, color='purple', label='R-M-TiTa', s=size)
        ax.scatter(*R_H_TiTa, color='pink', label='R-H-TiTa', s=size)
        
        ax.scatter(*L_F_FeTi, color='green', label='L-F-FeTi', s=size)
        ax.scatter(*L_M_FeTi, color='blue', label='L-M-FeTi', s=size)
        ax.scatter(*L_H_FeTi, color='red', label='L-H-FeTi', s=size)
        
        ax.scatter(*R_F_FeTi, color='yellow', label='R-F-FeTi', s=size)
        ax.scatter(*R_M_FeTi, color='purple', label='R-M-FeTi', s=size)
        ax.scatter(*R_H_FeTi, color='pink', label='R-H-FeTi', s=size)
        
        ax.scatter(*L_F_CTr, color='green', label='L-F-CTr', s=size)
        ax.scatter(*L_M_CTr, color='blue', label='L-M-CTr', s=size)
        ax.scatter(*L_H_CTr, color='red', label='L-H-CTr', s=size)
        
        ax.scatter(*R_F_CTr, color='yellow', label='R-F-CTr', s=size)
        ax.scatter(*R_M_CTr, color='purple', label='R-M-CTr', s=size)
        ax.scatter(*R_H_CTr, color='pink', label='R-H-CTr', s=size)
        
        ax.scatter(*L_F_ThC, color='green', label='L-F-ThC', s=size)
        ax.scatter(*L_M_ThC, color='blue', label='L-M-ThC', s=size)
        ax.scatter(*L_H_ThC, color='red', label='L-H-ThC', s=size)
        
        ax.scatter(*R_F_ThC, color='yellow', label='R-F-ThC', s=size)
        ax.scatter(*R_M_ThC, color='purple', label='R-M-ThC', s=size)
        ax.scatter(*R_H_ThC, color='pink', label='R-H-ThC', s=size)
        
        for i in L_F_stances:
            ax.scatter(*i, color='green', s=size+3, alpha=0.5)
        for i in L_M_stances:
            ax.scatter(*i, color='blue', s=size+3, alpha=0.5)
        for i in L_H_stances:
            ax.scatter(*i, color='red', s=size+3, alpha=0.5)
        for i in R_F_stances:
            ax.scatter(*i, color='yellow', s=size+3, alpha=0.5)
        for i in R_M_stances:
            ax.scatter(*i, color='purple', s=size+3, alpha=0.5)
        for i in R_H_stances:
            ax.scatter(*i, color='pink', s=size+3, alpha=0.5)
            

        if len(L_F_stances) > 50:
            L_F_stances.clear()
        if len(L_M_stances) > 50:
            L_M_stances.clear()
        if len(L_H_stances) > 50:
            L_H_stances.clear()
        if len(R_F_stances) > 50:
            R_F_stances.clear()
        if len(R_M_stances) > 50:
            R_M_stances.clear()
        if len(R_H_stances) > 50:
            R_H_stances.clear()

        
        ax.plot([L_F_TiTa[0], L_F[0]], [L_F_TiTa[1], L_F[1]], [L_F_TiTa[2], L_F[2]], color='darkorange')
        ax.plot([L_M_TiTa[0], L_M[0]], [L_M_TiTa[1], L_M[1]], [L_M_TiTa[2], L_M[2]], color='darkslategrey')
        ax.plot([L_H_TiTa[0], L_H[0]], [L_H_TiTa[1], L_H[1]], [L_H_TiTa[2], L_H[2]], color='darkmagenta')
        
        ax.plot([R_F_TiTa[0], R_F[0]], [R_F_TiTa[1], R_F[1]], [R_F_TiTa[2], R_F[2]], color='orangered')
        ax.plot([R_M_TiTa[0], R_M[0]], [R_M_TiTa[1], R_M[1]], [R_M_TiTa[2], R_M[2]], color='forestgreen')
        ax.plot([R_H_TiTa[0], R_H[0]], [R_H_TiTa[1], R_H[1]], [R_H_TiTa[2], R_H[2]], color='royalblue')
        
        ax.plot([L_F_TiTa[0], L_F_FeTi[0]], [L_F_TiTa[1], L_F_FeTi[1]], [L_F_TiTa[2], L_F_FeTi[2]], color='darkorange')
        ax.plot([L_M_TiTa[0], L_M_FeTi[0]], [L_M_TiTa[1], L_M_FeTi[1]], [L_M_TiTa[2], L_M_FeTi[2]], color='darkslategrey')
        ax.plot([L_H_TiTa[0], L_H_FeTi[0]], [L_H_TiTa[1], L_H_FeTi[1]], [L_H_TiTa[2], L_H_FeTi[2]], color='darkmagenta')
        
        ax.plot([R_F_TiTa[0], R_F_FeTi[0]], [R_F_TiTa[1], R_F_FeTi[1]], [R_F_TiTa[2], R_F_FeTi[2]], color='orangered')
        ax.plot([R_M_TiTa[0], R_M_FeTi[0]], [R_M_TiTa[1], R_M_FeTi[1]], [R_M_TiTa[2], R_M_FeTi[2]], color='forestgreen')
        ax.plot([R_H_TiTa[0], R_H_FeTi[0]], [R_H_TiTa[1], R_H_FeTi[1]], [R_H_TiTa[2], R_H_FeTi[2]], color='royalblue')
        
        ax.plot([L_F_CTr[0], L_F_FeTi[0]], [L_F_CTr[1], L_F_FeTi[1]], [L_F_CTr[2], L_F_FeTi[2]], color='darkorange')
        ax.plot([L_M_CTr[0], L_M_FeTi[0]], [L_M_CTr[1], L_M_FeTi[1]], [L_M_CTr[2], L_M_FeTi[2]], color='darkslategrey')
        ax.plot([L_H_CTr[0], L_H_FeTi[0]], [L_H_CTr[1], L_H_FeTi[1]], [L_H_CTr[2], L_H_FeTi[2]], color='darkmagenta')
        
        ax.plot([R_F_CTr[0], R_F_FeTi[0]], [R_F_CTr[1], R_F_FeTi[1]], [R_F_CTr[2], R_F_FeTi[2]], color='orangered')
        ax.plot([R_M_CTr[0], R_M_FeTi[0]], [R_M_CTr[1], R_M_FeTi[1]], [R_M_CTr[2], R_M_FeTi[2]], color='forestgreen')
        ax.plot([R_H_CTr[0], R_H_FeTi[0]], [R_H_CTr[1], R_H_FeTi[1]], [R_H_CTr[2], R_H_FeTi[2]], color='royalblue')
        
        ax.plot([L_F_CTr[0], L_F_ThC[0]], [L_F_CTr[1], L_F_ThC[1]], [L_F_CTr[2], L_F_ThC[2]], color='darkorange')
        ax.plot([L_M_CTr[0], L_M_ThC[0]], [L_M_CTr[1], L_M_ThC[1]], [L_M_CTr[2], L_M_ThC[2]], color='darkslategrey')
        ax.plot([L_H_CTr[0], L_H_ThC[0]], [L_H_CTr[1], L_H_ThC[1]], [L_H_CTr[2], L_H_ThC[2]], color='darkmagenta')
        
        ax.plot([R_F_CTr[0], R_F_ThC[0]], [R_F_CTr[1], R_F_ThC[1]], [R_F_CTr[2], R_F_ThC[2]], color='orangered')
        ax.plot([R_M_CTr[0], R_M_ThC[0]], [R_M_CTr[1], R_M_ThC[1]], [R_M_CTr[2], R_M_ThC[2]], color='forestgreen')
        ax.plot([R_H_CTr[0], R_H_ThC[0]], [R_H_CTr[1], R_H_ThC[1]], [R_H_CTr[2], R_H_ThC[2]], color='royalblue')
        
        r=df[df['fnum'] == frame]['r_ball'].mean()

        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = r*np.cos(u)*np.sin(v) + df[df['fnum'] == frame]['x_ball'].mean() 
        y = r*np.sin(u)*np.sin(v) + df[df['fnum'] == frame]['y_ball'].mean() 
        z = r*np.cos(v) + df[df['fnum'] == frame]['z_ball'].mean() 
        
        if ball == True:
            ax.plot_surface(x, y, z, color="navajowhite", alpha=0.4)

        
        ax.set_axis_off()
        
        if view == 'L':
            ax.view_init(90, 90, 0)
        if view == 'R':
            ax.view_init(-90, -90, 0)
        if view == 'T':
            ax.view_init(-120, 90, -90)

        frame_file = os.path.join(frames_path, f'frame_{frame:03d}.png')
        plt.savefig(frame_file, dpi=300)
        plt.close(fig)
        
        frame_files.append(frame_file)

   
    clip = ImageSequenceClip(frame_files, fps=40)
    video_path = os.path.join(path, f'{genotype}_fly{flynum}_trial{trial}_view{view}_ball{ball}.mp4')
    clip.write_videofile(video_path, codec='libx264')


    for frame_file in frame_files:
        os.remove(frame_file)
        
        
def plot_curvature(df, trial, P9RT, P9LT):
    
    L_F_stances = []
    L_M_stances = []
    L_H_stances = []
    R_F_stances = []
    R_M_stances = []
    R_H_stances = []
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    
    start = list(df['fnum'])[600*(trial-1)]
    end = start + 600
    fnums = range(start,end)
    
    for frame in tqdm(fnums):
        L_F = get_xyz_mean(df[df['fnum'] == frame], 'L-F-TaG')
        L_M = get_xyz_mean(df[df['fnum'] == frame], 'L-M-TaG')
        L_H = get_xyz_mean(df[df['fnum'] == frame], 'L-H-TaG')
        R_F = get_xyz_mean(df[df['fnum'] == frame], 'R-F-TaG')
        R_M = get_xyz_mean(df[df['fnum'] == frame], 'R-M-TaG')
        R_H = get_xyz_mean(df[df['fnum'] == frame], 'R-H-TaG')
            
        if df[df['fnum'] == frame]['L-F_stepcycle'][frame] == True:
            L_F_stances.append(L_F)
        if df[df['fnum'] == frame]['L-M_stepcycle'][frame] == True:
            L_M_stances.append(L_M)
        if df[df['fnum'] == frame]['L-H_stepcycle'][frame] == True:
            L_H_stances.append(L_H)
        if df[df['fnum'] == frame]['R-F_stepcycle'][frame] == True:
            R_F_stances.append(R_F)
        if df[df['fnum'] == frame]['R-M_stepcycle'][frame] == True:
            R_M_stances.append(R_M)
        if df[df['fnum'] == frame]['R-H_stepcycle'][frame] == True:
            R_H_stances.append(R_H)
            
    size=5
    
    if P9RT==True:
    
        for i in L_F_stances:
            ax.scatter(*i, color='green', s=size)
        for i in L_M_stances:
            ax.scatter(*i, color='blue', s=size)
        for i in L_H_stances:
            ax.scatter(*i, color='red', s=size)
 
        ax.view_init(180, 90, 90)
            
    if P9LT==True:
        
        for i in R_F_stances:
            ax.scatter(*i, color='yellow', s=size)
        for i in R_M_stances:
            ax.scatter(*i, color='purple', s=size)
        for i in R_H_stances:
            ax.scatter(*i, color='pink', s=size)
            
        ax.view_init(180, 90, 90)
        
    plt.show()
            
        
        

def plot_trajectory(df, genotype, flynum, trial, path):

    frames_path = os.path.join(path, 'frames')
    os.makedirs(frames_path, exist_ok=True)

    frame_files = []

    start = list(df['fnum'])[1400*(trial-1)]
    end = start + 1400
    fnums = range(start,end)
    
    trajectories = []
    

    
    for frame in tqdm(fnums):
        
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set_axis_off()
        
        
        trajectory = df['z_pos'][frame],df['y_pos'][frame]
        
        trajectories.append(trajectory)
        

        
        for i in trajectories:
            ax.scatter(*i, color='blue', alpha=0.25)

        
        
        
        frame_file = os.path.join(frames_path, f'frame_{frame:03d}.png')
        plt.savefig(frame_file, dpi=300)
        plt.close(fig)
        
        frame_files.append(frame_file)

   
    clip = ImageSequenceClip(frame_files, fps=40)
    video_path = os.path.join(path, f'{genotype}_fly{flynum}_trial{trial}_trajectory.mp4')
    clip.write_videofile(video_path, codec='libx264')
    
    for frame_file in frame_files:
        os.remove(frame_file)