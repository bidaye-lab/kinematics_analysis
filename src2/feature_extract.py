import os
import numpy as np
import glob as glob
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math
import seaborn as sns
import scipy
from scipy.optimize import minimize
from matplotlib.colors import Normalize

## define the column indices of the corresponding TaG coordinates
L1_TaG_idx = [61, 62, 63]
R1_TaG_idx = [16, 17, 18]
L2_TaG_idx = [76, 77, 78]
R2_TaG_idx = [31, 32, 33]
L3_TaG_idx = [91, 92, 93]
R3_TaG_idx = [46, 47, 48]


def get_TD_LO(df, Leg, idx_ts):
    """Extracts touchdown and liftoff frame indices from genotype dataframe using step cycle predictions

    Parameters
    ----------
    df : DataFrame
        Genotype dataframe with step cycle predictions
    Leg : string
        'L1'/'L2'/'L3' etc

    Returns
    -------
    DataFrame
        df with TD and LO time indices as columns or empty df if no TD or LO events detected
    """

    if Leg == "L1":
        SC = df["L-F_stepcycle"].astype(int)
    elif Leg == "L2":
        SC = df["L-M_stepcycle"].astype(int)
    elif Leg == "L3":
        SC = df["L-H_stepcycle"].astype(int)
    elif Leg == "R1":
        SC = df["R-F_stepcycle"].astype(int)
    elif Leg == "R2":
        SC = df["R-M_stepcycle"].astype(int)
    elif Leg == "R3":
        SC = df["R-H_stepcycle"].astype(int)

    Leg_diff = pd.DataFrame(np.diff(np.array(SC)))
    Leg_TD_frame = pd.DataFrame(Leg_diff.index[Leg_diff[0] == 1])
    Leg_LO_frame = pd.DataFrame(Leg_diff.index[Leg_diff[0] == -1])

    if (len(Leg_TD_frame) > 0) & (len(Leg_LO_frame) > 0):

        Leg_SC_init = pd.concat([Leg_TD_frame, Leg_LO_frame], axis=1)
        Leg_SC_init.columns = ["TD", "LO"]
        return Leg_SC_init + idx_ts
    else:
        # raise TypeError("INFO: No touchdown-liftoff pair detected in dataframe")
        Leg_SC_init = pd.DataFrame()
        return Leg_SC_init


# align arrays to get stance
def align_arr(SC_df):
    """Pairs up TD and LO events such that every TD is followd by a LO to form a complete stance event

    Parameters
    ----------
    SC_df : DataFrame
        Dataframe with TD and LO time indices

    Returns
    -------
    DataFrame
        Dataframe with paired TD-LO events, empty Df if input df was empty
    """
    if SC_df.empty:
        return pd.DataFrame()
    else:
        SC_df_new = pd.DataFrame()
        ## get first TD and LO events
        TD_1 = int(SC_df["TD"][0])
        LO_1 = int(SC_df["LO"][0])

        if LO_1 < TD_1:  # if the first event is a lift off
            # LO_new = SC_df.iloc[1:, 1].reset_index(drop=True)
            # SC_df_new = pd.concat([SC_df.iloc[:, 0], LO_new], axis=1).dropna()
            SC_df_new = pd.concat([SC_df["LO"], SC_df["TD"]], axis=1)
            # SC_df_new.columns = ["TD", "LO"]
        else:
            SC_df_new = SC_df

        return SC_df_new


def diff_row(df):
    diff_list = []
    for i in range(len(df)):
        temp_diff = np.diff(np.array(df.iloc[i]))
        diff_list.append(temp_diff[0])
    return pd.DataFrame(diff_list).reset_index(drop=True)


def diff_col(df):
    return pd.DataFrame(np.diff(np.array(df.iloc[:, 0]))).reset_index(drop=True)


def diff_next_step(df):
    diff_list = []
    for i in range(len(df) - 1):
        event1 = df.iloc[i, 1]
        event2 = df.iloc[i + 1, 0]
        temp_diff = event2 - event1
        diff_list.append(temp_diff.astype(float))
    return pd.DataFrame(diff_list).reset_index(drop=True)


def get_temp_params_2(leg_steps_velfilt):
    """Extracts temporal stepping parametrs (stance duration, swing duration, step period)
    using paired TD-LO indices

    Parameters
    ----------
    leg_steps_velfilt : DataFrame
         df with aligned TD-LO time indices

    Returns
    -------
    DataFrame
        appends columns with stepping parameters to input df, returns empty df if input was empty
    """
    if leg_steps_velfilt.empty:
        return pd.DataFrame()
    else:
        if leg_steps_velfilt.columns[0] == "LO":

            swing_dur = diff_row(leg_steps_velfilt)
            try:
                swing_dur.columns = ["swing_dur"]
            except:
                swing_dur = pd.DataFrame(columns=["swing_dur"])

            stance_dur = diff_next_step(leg_steps_velfilt)
            try:
                stance_dur.columns = ["stance_dur"]
            except:
                stance_dur = pd.DataFrame(columns=["stance_dur"])

            step_period = diff_col(leg_steps_velfilt)
            try:
                step_period.columns = ["step_period"]
            except:
                step_period = pd.DataFrame(columns=["step_period"])

        else:
            swing_dur = diff_next_step(leg_steps_velfilt)
            try:
                swing_dur.columns = ["swing_dur"]
            except:
                swing_dur = pd.DataFrame(columns=["swing_dur"])
            stance_dur = diff_row(leg_steps_velfilt)
            try:
                stance_dur.columns = ["stance_dur"]
            except:
                stance_dur = pd.DataFrame(columns=["stance_dur"])
            step_period = diff_col(leg_steps_velfilt)
            try:
                step_period.columns = ["step_period"]
            except:
                step_period = pd.DataFrame(columns=["step_period"])

        new_velfilt_df = pd.concat(
            [leg_steps_velfilt, step_period, swing_dur, stance_dur], axis=1
        )
        return new_velfilt_df


def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))


def get_heading_direction_vector(df):
    """Get the heading direction vector; deifned as the vector joining the notumm to the mid point of thee line joining the wing hinges

    Parameters
    ----------
    df : DataFrame
        Raw data table with position coordinates of all tracked points

    Returns
    -------
    tuple
        line segment defining the forward direction of the fly
    """

    x_mid = (np.mean(df["L-WH_x"]) + np.mean(df["R-WH_x"])) / 2
    y_mid = (np.mean(df["L-WH_y"]) + np.mean(df["R-WH_y"])) / 2
    # z_mid = (np.mean(df["L-WH_z"]) + np.mean(df["R-WH_z"])) / 2

    # ref_WH = (x_mid, y_mid, z_mid)
    ref_WH = (x_mid, y_mid)

    Ref_vector = (
        np.mean(df["Notum_x"]) - ref_WH[0],
        np.mean(df["Notum_y"]) - ref_WH[1],
        # np.mean(df["Notum_z"]) - ref_WH[2],
    )
    return Ref_vector


def get_stepping_direction(Ref_vector, TD_coords, LO_coords):
    """Returns the angle of each step with respect to the heading direction of the for each stance trajectory

    Parameters
    ----------
    Ref_vector : tuple
        _description_
    stepdata : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    test = (
        LO_coords.iloc[0, 0] - TD_coords.iloc[0, 0],
        LO_coords.iloc[0, 1] - TD_coords.iloc[0, 1],
        # LO_coords.iloc[0, 2] - TD_coords.iloc[0, 2]
    )
    angle = np.degrees(
        np.arccos(
            (np.dot(Ref_vector, test) / (magnitude(Ref_vector) * magnitude(test)))
        )
    )

    return angle


def get_stance_dist2(leg_steps_velfilt, data, leg, BL=1):
    """Quantifies and appends stance distance to input df

    Parameters
    ----------
    leg_steps_velfilt : DataFrame
        Datafram with paired TD-LO columns
    data : DataFrame
        original dataframe with location coordinates
    leg : string
        'L1'/'L2'/'L3' etc
    BL : int, optional
        body length of individual fly calculated as the distance between the two wing hinges, by default 1

    Returns
    -------
    DataFrame
        appends normalized stance distance to the input datarame; returns empty if input was empty
    """
    if leg_steps_velfilt.empty:
        return pd.DataFrame()
    else:
        if leg == "L1":
            Leg_TaG = data.iloc[:, L1_TaG_idx]
        if leg == "L2":
            Leg_TaG = data.iloc[:, L2_TaG_idx]
        if leg == "L3":
            Leg_TaG = data.iloc[:, L3_TaG_idx]
        if leg == "R1":
            Leg_TaG = data.iloc[:, R1_TaG_idx]
        if leg == "R2":
            Leg_TaG = data.iloc[:, R2_TaG_idx]
        if leg == "R3":
            Leg_TaG = data.iloc[:, R3_TaG_idx]

        stance_dist = []

        if leg_steps_velfilt.columns[0] == "TD":  ## if first event is stance
            if len(leg_steps_velfilt.dropna()) > 0:
                for step in range(len(leg_steps_velfilt.dropna())):
                    TD_pos_step = pd.DataFrame(
                        Leg_TaG.iloc[int(leg_steps_velfilt["TD"][step]), :]
                    ).T
                    LO_pos_step = pd.DataFrame(
                        Leg_TaG.iloc[int(leg_steps_velfilt["LO"][step]), :]
                    ).T
                    stance_dist_step = math.dist(
                        TD_pos_step.iloc[0, :], LO_pos_step.iloc[0, :]
                    )
                    stance_dist.append(stance_dist_step)
            else:
                stance_dist.append(np.nan)

        elif leg_steps_velfilt.columns[0] == "LO":  ## if first event is swing
            if len(leg_steps_velfilt) > 1:  ## if there are at least 2 lift offs
                for step in range(len(leg_steps_velfilt) - 1):
                    TD_pos_step = pd.DataFrame(
                        Leg_TaG.iloc[int(leg_steps_velfilt["TD"][step]), :]
                    ).T
                    LO_pos_step = pd.DataFrame(
                        Leg_TaG.iloc[int(leg_steps_velfilt["LO"][step + 1]), :]
                    ).T

                    stance_dist_step = math.dist(
                        TD_pos_step.iloc[0, :], LO_pos_step.iloc[0, :]
                    )
                    stance_dist.append(stance_dist_step)
            else:
                stance_dist.append(np.nan)

        else:
            raise TypeError("Unexpected input to function")

        # print((leg_steps_velfilt.columns[0], len(leg_steps_velfilt), stance_dist))
        stance_dist_norm = pd.DataFrame([elem / BL for elem in stance_dist])
        # stance_dist_norm.columns = ["stance_dist_norm"]

        Leg_all_data = pd.concat([leg_steps_velfilt, stance_dist_norm], axis=1)
        Leg_all_data = Leg_all_data.iloc[
            :, 2:
        ]  # dropping TD, LO columns in the beginning
        Leg_all_data.columns = [
            leg + "_step_period",
            leg + "_swing_dur",
            leg + "_stance_dur",
            leg + "_stance_dist_norm",
        ]

        return Leg_all_data


def get_ang_params(temp_win, leg):
    """Parmeterizes the joint angle traces using peak prominence, height and width

    Parameters
    ----------
    temp_win : dataframe
        original dataframe with joint angle time series
    leg : string
        'L1'/'L2'/'L3' etc

    Returns
    -------
    dataframe
        df with columns reporting the count of peaks, peak heights, widths and prominences for each joint angle
    """

    ang_name_dict = {}

    ang_list = ["A_flex", "B_flex", "C_flex", "A_rot", "B_rot", "C_rot", "A_abduct"]

    for ang in ang_list:
        if np.mean(temp_win[leg + ang]) > 0:
            height = 10
        else:
            height = -180
        ang_name_dict[leg + ang] = scipy.signal.find_peaks(
            temp_win[leg + ang], prominence=0.5, height=height, width=2
        )

    all_ang_data = pd.DataFrame()

    for peak in ang_name_dict.keys():
        count = len(ang_name_dict[peak][0])
        count_list = np.repeat(count, count)

        heights = ang_name_dict[peak][1]["peak_heights"]
        widths = ang_name_dict[peak][1]["widths"]
        prominences = ang_name_dict[peak][1]["prominences"]

        peak_all_data = pd.DataFrame([count_list, heights, widths, prominences]).T
        peak_all_data.columns = [
            peak + "_count",
            peak + "_heights",
            peak + "_widths",
            peak + "_prominences",
        ]
        all_ang_data = pd.concat([all_ang_data, peak_all_data], axis=1)
    return all_ang_data


def check_if_empty(stepdata_df, leg):
    """appends empty columns with the appripriate names in case there were no complete stepping events detected

    Parameters
    ----------
    stepdata_df : DataFrame
        steppping data df , either empty or with the relevant columns
    leg : string
        'L1'/'L2'/'L3' etc

    Returns
    -------
    DataFrame
        returns the input dtaaframe in case it was not empty;
        empty DataFrame with correct column names in case input was empty
    """
    if stepdata_df.empty:
        return pd.DataFrame(
            columns=[
                leg + "_step_period",
                leg + "_swing_dur",
                leg + "_stance_dur",
                leg + "_stance_dist_norm",
            ]
        )
    else:
        return stepdata_df


def gaussian_weights(window_size, sigma):
    """
    Generates Gaussian weights centered around the middle of the window_size.

    Parameters:
    - window_size (int): Size of the moving window.
    - sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
    - weights (ndarray): 1D array of Gaussian weights.
    """
    positions = np.arange(window_size)
    weights = np.exp(-((positions - window_size // 2) ** 2) / (2 * sigma**2))
    weights /= np.sum(weights)  # Normalize weights to sum to 1
    return weights


def get_gaussian_mean(arr, weights_temp):

    temp_mean = np.average(arr, weights=weights_temp)

    return temp_mean


def smoothed_table(genotype, window, sigma=1):
    """Constructs a DataFrame with stepping parameters (step cycle based and joint angle based),
    averaged over a moving window of the chisen size (in frames, 1 frame = 5ms)

    Parameters
    ----------
    genotype : DataFrame
        original datastructure with 3d pose, 3d angles, ball velocity and step cycle predictions
    window : int
        moving avergae bin size in frames (1 frame = 5ms)

    Returns
    -------
    DataFrame
        Contains the mean x_vel, mean z_vel, step cycle dependent stepping parameters,
        joint angle based parameters for all 6 legs.
    """
    window = int(window)
    DataSt = pd.DataFrame()
    weights_norm = gaussian_weights(window, sigma)

    flies = genotype["flynum"].unique()
    for fly in flies:
        trial = genotype.groupby("flynum").get_group(fly)["tnum"].unique()
        for t in trial:
            print("INFO: Extracting data for ", (fly, t))
            data = (
                genotype.groupby("flynum")
                .get_group(fly)
                .groupby("tnum")
                .get_group(t)
                .reset_index(drop=True)
            )

            data_stim = data.iloc[
                int(window / 2) : 1400, :
            ]  ## change to include only stim frames or just pre-stim period

            # Defining body length as distance between wing hinges
            BL = math.dist(
                [
                    np.mean(data["R-WH_x"]),
                    np.mean(data["R-WH_y"]),
                    np.mean(data["R-WH_z"]),
                ],
                [
                    np.mean(data["L-WH_x"]),
                    np.mean(data["L-WH_y"]),
                    np.mean(data["L-WH_z"]),
                ],
            )

            for i in range(
                len(data_stim) - window
            ):  ## total length = 1400 - (window*1.5)
                temp_win = data_stim.iloc[i : i + window, :]
                idx_ts = temp_win.index[0]

                mean_z_vel = get_gaussian_mean(
                    temp_win["z_vel"], weights_norm
                )  # np.mean((temp_win["z_vel"]))
                mean_y_vel = get_gaussian_mean(
                    temp_win["y_vel"], weights_norm
                )  # np.mean((temp_win["y_vel"]))
                mean_x_vel = get_gaussian_mean(
                    temp_win["x_vel"], weights_norm
                )  # np.mean(temp_win["x_vel"])

                vel_df = pd.DataFrame([fly, t, mean_x_vel, mean_y_vel, mean_z_vel]).T
                vel_df.columns = [
                    "flynum",
                    "tnum",
                    "mean_x_vel",
                    "mean_y_vel",
                    "mean_z_vel",
                ]

                L1_stepdata = get_stance_dist2(
                    get_temp_params_2(align_arr(get_TD_LO(temp_win, "L1", idx_ts))),
                    data,
                    "L1",
                    BL,
                )
                L2_stepdata = get_stance_dist2(
                    get_temp_params_2(align_arr(get_TD_LO(temp_win, "L2", idx_ts))),
                    data,
                    "L2",
                    BL,
                )
                L3_stepdata = get_stance_dist2(
                    get_temp_params_2(align_arr(get_TD_LO(temp_win, "L3", idx_ts))),
                    data,
                    "L3",
                    BL,
                )
                R1_stepdata = get_stance_dist2(
                    get_temp_params_2(align_arr(get_TD_LO(temp_win, "R1", idx_ts))),
                    data,
                    "R1",
                    BL,
                )
                R2_stepdata = get_stance_dist2(
                    get_temp_params_2(align_arr(get_TD_LO(temp_win, "R2", idx_ts))),
                    data,
                    "R2",
                    BL,
                )
                R3_stepdata = get_stance_dist2(
                    get_temp_params_2(align_arr(get_TD_LO(temp_win, "R3", idx_ts))),
                    data,
                    "R3",
                    BL,
                )

                L1_stepdata = check_if_empty(L1_stepdata, "L1")
                L2_stepdata = check_if_empty(L2_stepdata, "L2")
                L3_stepdata = check_if_empty(L3_stepdata, "L3")
                R1_stepdata = check_if_empty(R1_stepdata, "R1")
                R2_stepdata = check_if_empty(R2_stepdata, "R2")
                R3_stepdata = check_if_empty(R3_stepdata, "R3")

                L1_angdata = get_ang_params(temp_win, "L1")
                L2_angdata = get_ang_params(temp_win, "L2")
                L3_angdata = get_ang_params(temp_win, "L3")
                R1_angdata = get_ang_params(temp_win, "R1")
                R2_angdata = get_ang_params(temp_win, "R2")
                R3_angdata = get_ang_params(temp_win, "R3")

                temp_win_data = pd.concat(
                    [
                        L1_stepdata,
                        L1_angdata,
                        L2_stepdata,
                        L2_angdata,
                        L3_stepdata,
                        L3_angdata,
                        R1_stepdata,
                        R1_angdata,
                        R2_stepdata,
                        R2_angdata,
                        R3_stepdata,
                        R3_angdata,
                    ],
                    axis=1,
                ).mean()

                temp_win_data = pd.concat(
                    [vel_df, pd.DataFrame(temp_win_data).T], axis=1
                )

                ## replacing NaNs by zeros
                ## for V1 replacing anythiung that is an Nan by zero. Future version to use more sensible values
                for col in temp_win_data.columns:
                    temp_win_data[col] = temp_win_data[col].fillna(0)

                DataSt = pd.concat([DataSt, temp_win_data], axis=0, ignore_index=True)
    return DataSt
