# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: kine
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import pandas as pd

from src import (
    data_loader as dl,
    df_operations as dfo,
    fitting as fit,
    visualize as vis,
)

# %% [markdown]
# # Run ball fitting in batch mode

# %%
# load data
cfg = dl.load_config('config.yml')
data = dl.load_data_hdf(cfg['datafile'])

# define percentiles for each leg
d_perc = {
    'R-F':  (25, 75),
    'R-M':  (25, 75),
    'R-H':  (25, 75),
    'L-F':  (25, 75),
    'L-M':  (25, 75),
    'L-H':  (25, 75),
}
    
# thresholds for step detection
min_on, min_off = 2, 2 # mimimum number of frames for on/off step 
d_delta_r = { # distance from median per leg (unit?)
    'R-F': .05,
    'R-M': .05,
    'R-H': .05,
    'L-F': .05,
    'L-M': .05,
    'L-H': .05,
}
# data frame for ball centers / radii
df_ball = pd.DataFrame()

# cycle trough genotypes
idx = 0
for gen, df_gen in data.items():

    # plot folder
    plot_folder = Path(cfg['plot_folder']) / 'ball_predictions/{}/'.format(gen)
    plot_folder.mkdir(parents=True, exist_ok=True)

    # cycle through flies
    for fly, df_fly in df_gen.groupby('flynum'):

        print('INFO: processing genotype {} | fly {}'.format(gen, fly))
        print('      ==================='.format(gen, fly))
        
        #######################
        ## only use stim frames
        df = dfo.filter_frames(df_fly)

        # fit ball
        ball, r = fit.fit_ball(df, d_perc)
        print('Optimized: ball center x = {:1.3f} y = {:1.3f} z = {:1.3f} | radius {:1.3f}'.format(*ball, r))

        # add distances from center 
        df = dfo.add_distance(df, ball)

        # get "median" for TaG_r for each leg
        d_med = dfo.get_r_median(df, d_perc)

        # write to df_ball
        df_ball.loc[idx, 'genotype'] = gen
        df_ball.loc[idx, 'flynum'] = fly
        df_ball.loc[idx, ['ball_x', 'ball_y', 'ball_z']] = ball
        df_ball.loc[idx, 'r'] = r
        for k, v in d_perc.items():
            df_ball.at[idx, 'perc_low_{}'.format(k)] = v[0]
            df_ball.at[idx, 'perc_high_{}'.format(k)] = v[1]
        idx += 1

        #######################
        ## all frames

        # add distances from center 
        df_fly = dfo.add_distance(df_fly, ball)

        # step cycles
        df_fly = dfo.add_stepcycle_pred(df_fly, d_med, d_delta_r, min_on, min_off)

        # add back to data dict
        data[gen].loc[df_fly.index, df_fly.columns] = df_fly

        #####################
        ## plot for stim only
        df = dfo.filter_frames(df_fly)

        # plot r distribution
        vis.plot_r_distr(df, 'TaG_r', d_perc, path=plot_folder / 'r_distr_fly{}.png'.format(fly))
        
        # plot stepcycles 
        vis.plot_stepcycle_pred_grid(df, d_med, d_delta_r, path=plot_folder / 'stepcycles_{}.png'.format(fly))

# store on disk
path_df_ball = Path(cfg['data_folder']) / 'df_ball.parquet'
df_ball.to_parquet(path_df_ball)

out_file = Path(cfg['data_folder']) / 'df_preproc.parquet'
dl.write_data_dict(data, out_file)
