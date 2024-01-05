import yaml
from copy import deepcopy

from src import (
    data_loader as dl,
    df_operations as dfo,
    fitting as fit,
    visualize as vis,
)

def write_fit_params(params_dict, path):
    '''Write fit parameters to YAML file.

    This writes the parameter set used to create the plots
    to a YAML file. The parameters are doubled: 'old' and 'new'.
    The 'old' parameters are the ones used to create the plots.
    Changing the 'new' parameters will trigger reanalysis.

    Parameters
    ----------
    params_dict : dict
        Parameter set for each fly as well as global parameter set.
    path : Path
        File to save parameters to.
    '''

    # Write the dictionary to the YAML file
    with open(path, 'w') as file:

        for k, params in params_dict.items():
            
            old_new = {}
            for p, v in params.items():

                # create deepcopy to avoid aliases in YAML
                v_old = v
                v_new = deepcopy(v)
                old_new[p] = {'new': v_new, 'old': v_old }
                
            yaml.dump({k: old_new}, file, default_flow_style=None)
            file.write('\n')

def have_params_changed(path, key):
    '''Check if parameter set has changed.

    This compares for a given parameter set the 'old' and 'new' parameters. 

    Parameters
    ----------
    path : Path
        Path to YAML file with parameters.
    key : str
        Name of parameter set to check.

    Returns
    -------
    bool
        True if any parameter changed, False otherwise.
    '''
    
    params_dict = dl.load_config(path)
    params = params_dict[key]

    for v in params.values():

        p_old, p_new = v['old'], v['new'] 

        # if any parameter changed, run
        if p_old != p_new:
            return True
    
    # if no parameter changed, do not run
    return False


def load_params(path, key):
    '''Load the 'new' parameters from a YAML file.

    Parameters
    ----------
    path : Path
        Path to YAML file with parameters.
    key : str
        Name of parameter set to load.

    Returns
    -------
    params : dict
        Dictionary with only 'new' parameter set.
    '''

    params_disk = dl.load_config(path)
    old_new = params_disk[key]
    params = {k: v['new'] for k, v in old_new.items()}

    return params

def check_global(params_path, default_params):
    '''Check if complete analysis needs to be run.

    Parameters
    ----------
    params_path : Path
        Path to YAML file with parameters.
    default_params : dict
        Default parameter set if no YAML file exists.

    Returns
    -------
    run_global : bool
        True if complete analysis needs to be run, False otherwise.
    params_all : dict
        Dictionary with only 'global' parameter set.
    '''

    # define default parameters
    params_all = { 'global': default_params }
    
    if not params_path.exists():
        # if no params file exists, run everything
        run_global = True
    elif have_params_changed(params_path, 'global'):
        # if global params changed, run everything
        run_global = True
        # load params from file
        params_all['global'] = load_params(params_path, 'global')
    else:
        run_global = False
    
    return run_global, params_all


def fit_ball_wrapper(df, params):
    "Wrapper for ball fitting."

    f0, ff = params['fit_frames']
    pct_range = params['pct_range']
    d_delta_r = params['d_delta_r']
    min_on, min_off = params['min_on'], params['min_off']

    # filter frames for fitting
    df_fit = dfo.filter_frames(df, f0, ff)

    # fit ball
    ball, r = fit.fit_ball(df_fit, pct_range)
    print('Optimized: ball center x = {:1.3f} y = {:1.3f} z = {:1.3f} | radius {:1.3f}'.format(*ball, r))

    # add distances from center 
    df_fit = dfo.add_distance(df_fit, ball)

    # get "median" for TaG_r for each leg
    d_med = dfo.get_r_median(df_fit, pct_range)

    # add distances from center for all frames
    df = dfo.add_distance(df, ball)

    # step cycles
    df = dfo.add_stepcycle_pred(df, d_med, d_delta_r, min_on, min_off)

    return df, d_med

def plot_fit_results_wrapper(df, params, d_med, out_folder):
    "Wrapper for plotting fit results."
    
    f0, ff = params['fit_frames']
    d_delta_r = params['d_delta_r']
    pct_range = params['pct_range']

    for trial, df_trial in df.groupby('tnum'):

        # plot r distribution
        vis.plot_r_distr(df_trial, 'TaG_r', pct_range, path=out_folder / f'r_distr_trial_{trial}.png')
        
        # plot stepcycles 
        vis.plot_stepcycle_pred(df_trial, d_med, d_delta_r, vspan=(f0, ff), path=out_folder / f'stepcycles_trial_{trial}.png')

def refine_fit_wrapper(df, out_folder, params_path, default_params):
    "Wrapper for refining fit results."

    out_folder.mkdir(parents=True, exist_ok=True)

    run_global, params_all = check_global(params_path, default_params)

    # cycle through flies
    for fly, df_fly in df.groupby('flynum'):

        key = f'fly_{fly}'
        
        if run_global:
            run_fly = True
            params_fly = params_all['global']
        else:
            run_fly = have_params_changed(params_path, key)
            params_fly = load_params(params_path, key)
        params_all[key] = params_fly

        if not run_fly:
            print(f'INFO skipping fly {fly}')
            print(f'----')
            continue
        else:
            print(f'INFO processing fly {fly}')
            print(f'----')


        # output folder for fly
        out_fly = out_folder / f'fly_{fly}/'
        out_fly.mkdir(exist_ok=True)

        # fit ball
        df_fly, d_med = fit_ball_wrapper(df_fly, params_fly)

        # plot stepcycle predictions
        plot_fit_results_wrapper(df_fly, params_fly, d_med, out_fly)

    # store to disk
    write_fit_params(params_all, out_folder / 'fit_params.yml')

def add_stepcyles(df, params_path):
    '''Add stepcycles to dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with data for flies in `params_path`.
    params_path : Path
        Path to YAML file with parameters.
    '''
    
    # cycle through flies
    for fly, df_fly in df.groupby('flynum'):

        print(f'INFO processing fly {fly}')
        print(f'----')
        
        key = f'fly_{fly}'
        params_fly = load_params(params_path, key)

        df_fly, _ = fit_ball_wrapper(df_fly, params_fly)

        df.loc[df_fly.index, df_fly.columns] = df_fly

