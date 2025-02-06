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

def closest_point_and_distance(p, a, b):
    s = b - a
    w = p - a
    ps = np.dot(w, s)
    if ps <= 0:
        return a, np.linalg.norm(w)
    l2 = np.dot(s, s)
    if ps >= l2:
        closest = b
    else:
        closest = a + ps / l2 * s
    return closest
    

def get_xyz_mean(df, point):

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


def shortest_distance(pnt, a, b, c, d): 
    x1,y1,z1 = pnt
    d = abs((a * x1 + b * y1 + c * z1 + d)) 
    e = (math.sqrt(a * a + b * b + c * c))
    dist = d/e
    return dist


def distances(points, a, b, c, d):
    distances = []
    for point in points:
        x, y, z = point
        dist = np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
        distances.append(dist)
    return distances


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

    
    indices_R3B,_ = find_peaks(filtered_df['R3B_flex'], height=0, prominence=35)
    indices_R2B,_ = find_peaks(filtered_df['R2B_flex'], height=0, prominence=35)
    indices_R1B,_ = find_peaks(filtered_df['R1B_flex'], height=0, prominence=35)

    indices_L3B,_ = find_peaks(filtered_df['L3B_flex'], height=0,prominence=35)
    indices_L2B,_ = find_peaks(filtered_df['L2B_flex'], height=0, prominence=35)
    indices_L1B,_ = find_peaks(filtered_df['L1B_flex'], height=0, prominence=35)

 
    L1B_df = filtered_df.iloc[indices_L1B]
    L2B_df = filtered_df.iloc[indices_L2B]
    L3B_df = filtered_df.iloc[indices_L3B]
    R1B_df = filtered_df.iloc[indices_R1B]
    R2B_df = filtered_df.iloc[indices_R2B]
    R3B_df = filtered_df.iloc[indices_R3B]
    
    return L1B_df, L2B_df, L3B_df, R1B_df, R2B_df, R3B_df


def create_plane(df):

    notum = get_xyz_mean(df, 'Notum')
    lwh = get_xyz_mean(df, 'L-WH')
    rwh = get_xyz_mean(df, 'R-WH')

    
    L1B_df, L2B_df, L3B_df, R1B_df, R2B_df, R3B_df = joint_angle_filtering(df)

   
    L_F = get_xyz_mean(L1B_df, 'L-F-TaG')
    L_M = get_xyz_mean(L2B_df, 'L-M-TaG')
    L_H = get_xyz_mean(L3B_df, 'L-H-TaG')
    R_F = get_xyz_mean(R1B_df, 'R-F-TaG')
    R_M = get_xyz_mean(R2B_df, 'R-M-TaG')
    R_H = get_xyz_mean(R3B_df, 'R-H-TaG')


    tag_points = np.array([L_F, L_M, R_F, R_M, L_H, R_H])
    tag_centroid = tag_points.mean(axis=0)

    
    v1 = R_F - L_F
    v2 = tag_centroid - L_F

  
    normal_vector = np.cross(v1, v2)
    normal_vector /= np.linalg.norm(normal_vector)

    
    a, b, c = normal_vector
    d = -np.dot(normal_vector, L_F)  

    return a, b, c, d


def cost_fun_plane(x, l_pnts, l_perc):
    a, b, c, d = x

    cost = 0
    for pnts, perc in zip(l_pnts, l_perc):
        r = distances(pnts, a, b, c, d)
        r = np.array(r)

        
        lower, upper = np.nanpercentile(r, perc)
        r = r[(r > lower) & (r < upper)]

        
        cost += np.sum((r)**2)

    return cost


def fit_plane(df):
    d_perc={'R-F': [0, 100],'R-M': [0, 100],'R-H': [0, 100],'L-F': [0, 100],'L-M': [0, 100],'L-H': [0, 100]}
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

    # Extract percentages
    cols_x = [c for c in df.columns if 'TaG_x' in c]
    l_perc = [d_perc[c_x[:3]] for c_x in cols_x]

    # Get initial guess for the plane
    x0 = create_plane(df)

    # Optimize the cost function
    res = minimize(cost_fun_plane, x0, args=(l_pnts, l_perc), method='Nelder-Mead')
    plane = res.x
    return plane


def add_distance(df, plane):

    df = df.copy()

    # select all columns ending with x, y or z

    # cycle through all columns ending with `_x``
    cols_x = [c for c in df.columns if c[-2:] == '_x' ]
    for c_x in cols_x:

        # corresponding `_y` and `_z` columns
        c_y, c_z = '{}y'.format(c_x[:-1]), '{}z'.format(c_x[:-1])

        # calculate distance
        coords = df.loc[:, [c_x, c_y, c_z]].values
        a,b,c,d = plane
        dist = distances(coords,a,b,c,d)

        # write to df 
        r = '{}_r'.format(c_x[:-2])
        df.loc[:, r] = dist
    
    return df


def add_stepcycle_pred(df, r_med, min_on, min_off):
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
    lwh = get_xyz_mean(df, 'L-WH')
    rwh = get_xyz_mean(df, 'R-WH')
    dwh = np.linalg.norm(lwh - rwh)
    
    r_med = {'R-F': r_med['R-F']*dwh, 'R-M': r_med['R-M']*dwh, 'R-H': r_med['R-H']*dwh, 'L-F': r_med['L-F']*dwh, 'L-M': r_med['L-M']*dwh, 'L-H': r_med['L-H']*dwh}
    d_delta_r = {'R-F': 0.0, 'R-M': 0.0, 'R-H': 0.0, 'L-F': 0.0, 'L-M': 0.0, 'L-H': 0.0}

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


#----------------------------------------------------------------------------------------------------------------------------
#Quality Control


def step_cycle_plots(df, flynum, trial, angle, save, genotype, path):
    fig, axs = plt.subplots(2, 3, figsize=(20,10)) 
    axs[0,0].plot(df[f'R1{angle}_flex'][(1400*(trial-1)):(1400*trial)])
    axs[0,0].plot(df['R-F_stepcycle'][(1400*(trial-1)):(1400*trial)]*160)
    axs[0,0].title.set_text("R-F")

    axs[0,1].plot(df[f'R2{angle}_flex'][(1400*(trial-1)):(1400*trial)])
    axs[0,1].plot(df['R-M_stepcycle'][(1400*(trial-1)):(1400*trial)]*160)
    axs[0,1].title.set_text("R-M")

    axs[0,2].plot(df[f'R3{angle}_flex'][(1400*(trial-1)):(1400*trial)])
    axs[0,2].plot(df['R-H_stepcycle'][(1400*(trial-1)):(1400*trial)]*160)
    axs[0,2].title.set_text("R-H")

    axs[1,0].plot(df[f'L1{angle}_flex'][(1400*(trial-1)):(1400*trial)])
    axs[1,0].plot(df['L-F_stepcycle'][(1400*(trial-1)):(1400*trial)]*160)
    axs[1,0].title.set_text("L-F")

    axs[1,1].plot(df[f'L2{angle}_flex'][(1400*(trial-1)):(1400*trial)])
    axs[1,1].plot(df['L-M_stepcycle'][(1400*(trial-1)):(1400*trial)]*160)
    axs[1,1].title.set_text("L-M")

    axs[1,2].plot(df[f'L3{angle}_flex'][(1400*(trial-1)):(1400*trial)])
    axs[1,2].plot(df['L-H_stepcycle'][(1400*(trial-1)):(1400*trial)]*160)
    axs[1,2].title.set_text("L-H")
    
    
    if save == True:
        
        
        step_cycle_path = os.path.join(path, 'step_cycle_plots')
        os.makedirs(step_cycle_path, exist_ok=True)
        
        os.chdir(step_cycle_path)
        
        plt.savefig(f"fly{flynum}, trial{trial}.png", transparent=True)
    

    
    
def distance_from_surface_plots(df, r_med, flynum, trial, size, save, genotype, path):
    lwh = get_xyz_mean(df, 'L-WH')
    rwh = get_xyz_mean(df, 'R-WH')
    dwh = np.linalg.norm(lwh - rwh)
    
    fig, axs = plt.subplots(2, 3, figsize=(17,10), sharey=True) 

    axs[0,0].scatter(np.arange(1400), df['R-F-TaG_r'][(1400*(trial-1)):(1400*trial)], s=size)
    axs[0,0].axhline(y = r_med['R-F']*dwh, color = 'r', linestyle = '--')
    axs[0,0].axhline(y = 0*dwh, color = 'r', linestyle = '-')
    axs[0,0].title.set_text("R-F")


    axs[0,1].scatter(np.arange(1400), df['R-M-TaG_r'][(1400*(trial-1)):(1400*trial)], s=size)
    axs[0,1].axhline(y = r_med['R-M']*dwh, color = 'r', linestyle = '--')
    axs[0,1].axhline(y = 0*dwh, color = 'r', linestyle = '-')
    axs[0,1].title.set_text("R-M")


    axs[0,2].scatter(np.arange(1400), df['R-H-TaG_r'][(1400*(trial-1)):(1400*trial)], s=size)
    axs[0,2].axhline(y = r_med['R-H']*dwh, color = 'r', linestyle = '--')
    axs[0,2].axhline(y = 0*dwh, color = 'r', linestyle = '-')
    axs[0,2].title.set_text("R-H")


    axs[1,0].scatter(np.arange(1400), df['L-F-TaG_r'][(1400*(trial-1)):(1400*trial)], s=size)
    axs[1,0].axhline(y = r_med['L-F']*dwh, color = 'r', linestyle = '--')
    axs[1,0].axhline(y = 0*dwh, color = 'r', linestyle = '-')
    axs[1,0].title.set_text("L-F")


    axs[1,1].scatter(np.arange(1400), df['L-M-TaG_r'][(1400*(trial-1)):(1400*trial)], s=size)
    axs[1,1].axhline(y = r_med['L-M']*dwh, color = 'r', linestyle = '--')
    axs[1,1].axhline(y = 0*dwh, color = 'r', linestyle = '-')
    axs[1,1].title.set_text("L-M")


    axs[1,2].scatter(np.arange(1400), df['L-H-TaG_r'][(1400*(trial-1)):(1400*trial)], s=size)
    axs[1,2].axhline(y = r_med['L-H']*dwh, color = 'r', linestyle = '--')
    axs[1,2].axhline(y = 0*dwh, color = 'r', linestyle = '-')
    axs[1,2].title.set_text("L-H")

    plt.tight_layout()

    
    if save == True:
        
        
        distance_path = os.path.join(path, 'distance_plots')
        os.makedirs(distance_path, exist_ok=True)
        
        os.chdir(distance_path)
        
        plt.savefig(f"fly{flynum}, trial{trial}.png", transparent=True)
        


def plot_plane(df, plane_coefficients):
    a, b, c, d = plane_coefficients

    notum = get_xyz_mean(df, 'Notum')
    lwh = get_xyz_mean(df, 'L-WH')
    rwh = get_xyz_mean(df, 'R-WH')
    
    L1B_df, L2B_df, L3B_df, R1B_df, R2B_df, R3B_df = joint_angle_filtering(df)

    L_F = get_xyz_mean(L1B_df, 'L-F-TaG')
    L_M = get_xyz_mean(L2B_df, 'L-M-TaG')
    L_H = get_xyz_mean(L3B_df, 'L-H-TaG')
    R_F = get_xyz_mean(R1B_df, 'R-F-TaG')
    R_M = get_xyz_mean(R2B_df, 'R-M-TaG')
    R_H = get_xyz_mean(R3B_df, 'R-H-TaG')

    tag_points = np.array([L_F, L_M, R_F, R_M, L_H, R_H])
    tag_centroid = tag_points.mean(axis=0)

    L_F_ThC = get_xyz_mean(df, 'L-F-ThC')
    L_M_ThC = get_xyz_mean(df, 'L-M-ThC')
    L_H_ThC = get_xyz_mean(df, 'L-H-ThC')
    
    R_F_ThC = get_xyz_mean(df, 'R-F-ThC')
    R_M_ThC = get_xyz_mean(df, 'R-M-ThC')
    R_H_ThC = get_xyz_mean(df, 'R-H-ThC')

    x = np.linspace(lwh[0] - 1.25, rwh[0] + 1.25, 10)
    y = np.linspace(lwh[1] - 1.25, rwh[1] + 1.25, 10)
    x, y = np.meshgrid(x, y)
    
    if c != 0:
        z = (-a * x - b * y - d) / c
    else:
        z = np.zeros_like(x)

    fig = plt.figure()
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0)
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(x, y, z, alpha=0.15)

    ax.scatter(*L_F, color='green', label='L-F-TaG', marker='v')
    ax.scatter(*L_M, color='green', label='L-M-TaG')
    ax.scatter(*L_H, color='green', label='L-H-TaG', marker='s')
    
    ax.scatter(*R_F, color='orange', label='R-F-TaG', marker='v')
    ax.scatter(*R_M, color='orange', label='R-M-TaG')
    ax.scatter(*R_H, color='orange', label='R-H-TaG', marker='s')
    
    ax.scatter(*L_F_ThC, color='blue', label='L-F-ThC')
    ax.scatter(*L_M_ThC, color='blue', label='L-M-ThC')
    ax.scatter(*L_H_ThC, color='blue', label='L-H-ThC')
    
    ax.scatter(*R_F_ThC, color='purple', label='R-F-ThC')
    ax.scatter(*R_M_ThC, color='purple', label='R-M-ThC')
    ax.scatter(*R_H_ThC, color='purple', label='R-H-ThC')
    
    ax.plot([L_F_ThC[0], L_F[0]], [L_F_ThC[1], L_F[1]], [L_F_ThC[2], L_F[2]], color='blue')
    ax.plot([L_M_ThC[0], L_M[0]], [L_M_ThC[1], L_M[1]], [L_M_ThC[2], L_M[2]], color='blue')
    ax.plot([L_H_ThC[0], L_H[0]], [L_H_ThC[1], L_H[1]], [L_H_ThC[2], L_H[2]], color='blue')
    
    ax.plot([R_F_ThC[0], R_F[0]], [R_F_ThC[1], R_F[1]], [R_F_ThC[2], R_F[2]], color='purple')
    ax.plot([R_M_ThC[0], R_M[0]], [R_M_ThC[1], R_M[1]], [R_M_ThC[2], R_M[2]], color='purple')
    ax.plot([R_H_ThC[0], R_H[0]], [R_H_ThC[1], R_H[1]], [R_H_ThC[2], R_H[2]], color='purple')
    
    ax.view_init(elev = 50, azim = 0, roll=-190)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_axis_off()

    

    plt.show()
    
    
def plot_plane_gif(df, plane_coefficients, path, flynum, genotype):
    a, b, c, d = plane_coefficients
    

    frames_path = os.path.join(path, 'frames')
    os.makedirs(frames_path, exist_ok=True)

    notum = get_xyz_mean(df, 'Notum')
    lwh = get_xyz_mean(df, 'L-WH')
    rwh = get_xyz_mean(df, 'R-WH')
    
    L1B_df, L2B_df, L3B_df, R1B_df, R2B_df, R3B_df = joint_angle_filtering(df)

    L_F = get_xyz_mean(L1B_df, 'L-F-TaG')
    L_M = get_xyz_mean(L2B_df, 'L-M-TaG')
    L_H = get_xyz_mean(L3B_df, 'L-H-TaG')
    R_F = get_xyz_mean(R1B_df, 'R-F-TaG')
    R_M = get_xyz_mean(R2B_df, 'R-M-TaG')
    R_H = get_xyz_mean(R3B_df, 'R-H-TaG')

    tag_points = np.array([L_F, L_M, R_F, R_M, L_H, R_H])
    tag_centroid = tag_points.mean(axis=0)

    L_F_ThC = get_xyz_mean(df, 'L-F-ThC')
    L_M_ThC = get_xyz_mean(df, 'L-M-ThC')
    L_H_ThC = get_xyz_mean(df, 'L-H-ThC')
    R_F_ThC = get_xyz_mean(df, 'R-F-ThC')
    R_M_ThC = get_xyz_mean(df, 'R-M-ThC')
    R_H_ThC = get_xyz_mean(df, 'R-H-ThC')

    x = np.linspace(lwh[0] - 2.25, rwh[0] + 2.25, 10)
    y = np.linspace(lwh[1] - 2.25, rwh[1] + 2.25, 10)
    x, y = np.meshgrid(x, y)
    z = (-a * x - b * y - d) / c

    for angle in tqdm(range(0, 360, 2)):  # Rotate from 0 to 360 degrees
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot_surface(x, y, z, alpha=0.5)

        #ax.scatter(*notum, color='red', label='Notum')
        
        ax.scatter(*L_F, color='green', label='L-F-TaG', marker='v')
        ax.scatter(*L_M, color='green', label='L-M-TaG')
        ax.scatter(*L_H, color='green', label='L-H-TaG', marker='s')
        
        ax.scatter(*R_F, color='orange', label='R-F-TaG', marker='v')
        ax.scatter(*R_M, color='orange', label='R-M-TaG')
        ax.scatter(*R_H, color='orange', label='R-H-TaG', marker='s')
        
        #ax.scatter(*L_F_ThC, color='blue', label='L-F-ThC')
        #ax.scatter(*L_M_ThC, color='blue', label='L-M-ThC')
        #ax.scatter(*L_H_ThC, color='blue', label='L-H-ThC')
        
        #ax.scatter(*R_F_ThC, color='purple', label='R-F-ThC')
        #ax.scatter(*R_M_ThC, color='purple', label='R-M-ThC')
        #ax.scatter(*R_H_ThC, color='purple', label='R-H-ThC')
        
        #ax.plot([L_F_ThC[0], L_F[0]], [L_F_ThC[1], L_F[1]], [L_F_ThC[2], L_F[2]], color='blue')
        #ax.plot([L_M_ThC[0], L_M[0]], [L_M_ThC[1], L_M[1]], [L_M_ThC[2], L_M[2]], color='blue')
        #ax.plot([L_H_ThC[0], L_H[0]], [L_H_ThC[1], L_H[1]], [L_H_ThC[2], L_H[2]], color='blue')
        
        #ax.plot([R_F_ThC[0], R_F[0]], [R_F_ThC[1], R_F[1]], [R_F_ThC[2], R_F[2]], color='purple')
        #ax.plot([R_M_ThC[0], R_M[0]], [R_M_ThC[1], R_M[1]], [R_M_ThC[2], R_M[2]], color='purple')
        #ax.plot([R_H_ThC[0], R_H[0]], [R_H_ThC[1], R_H[1]], [R_H_ThC[2], R_H[2]], color='purple')
        
        #ax.scatter(*tag_centroid, color='black', label='Centroid')
        ax.view_init(elev=30, azim=angle)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_axis_off()

        plt.savefig(f'{frames_path}/frame_{angle:03d}.png')
        plt.close(fig)

    frames = [Image.open(f'{frames_path}/frame_{angle:03d}.png') for angle in range(0, 360, 2)]
    gif_path = os.path.join(path, f'{genotype}_fly{flynum}_3d_rotation.gif')
    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
    
    
def peak_plots(df, angle, h, p):
    filtered_parts = []
    for i in range(9):
        start_idx = 400 + (i * 1400)
        end_idx = start_idx + 600
        filtered_parts.append(df.iloc[start_idx:end_idx])

    filtered_df = pd.concat(filtered_parts)

    indices, properties = find_peaks(filtered_df[angle],height=h, prominence=p)

    peak_heights = properties['peak_heights']


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=filtered_df[angle],
        mode='lines',
        name='Original Plot'
    ))

    fig.add_trace(go.Scatter(
        x=indices,
        y= peak_heights,
        mode='markers',
        marker=dict(
        size=8,
        color='red',
        symbol='cross'
        ),
        name='Detected Peaks'
        ))

    fig.show()
    
    
    
def plot_walk_vid(df, path, flynum, genotype, trial):
    from tqdm import tqdm
    from moviepy.editor import ImageSequenceClip

    
    frames_path = os.path.join(path, 'frames')
    os.makedirs(frames_path, exist_ok=True)

    notum = get_xyz_mean(df, 'Notum')
    lwh = get_xyz_mean(df, 'L-WH')
    rwh = get_xyz_mean(df, 'R-WH')


    frame_files = []
    fnums = range(1400*(trial-1),1400*trial)
    
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

        fig = plt.figure(figsize=(10,8))
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
        
        ax.view_init(elev = -180, azim = 90, roll=-90)
        
        ax.set_axis_off()
        ax.set_zlim3d(86, 94)
        ax.set_xlim3d(-2, 2.5)
        
        frame_file = os.path.join(frames_path, f'frame_{frame:03d}.png')
        plt.savefig(frame_file, dpi=300)
        plt.close(fig)
        
        frame_files.append(frame_file)
   
    clip = ImageSequenceClip(frame_files, fps=40)
    video_path = os.path.join(path, f'{genotype}_fly{flynum}_trial{trial}.mp4')
    clip.write_videofile(video_path, codec='libx264')


    for frame_file in frame_files:
        os.remove(frame_file)