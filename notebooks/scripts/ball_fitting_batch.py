# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

from src import (
    data_loader as dl,
    batch_helpers as bh,
)
from src.xyz_trajectory import generate_xyz as xyz

# %% [markdown]
# # Run ball fitting in batch mode
#
# First, we load the data structure, which is a dictionary of DataFrames.
#
# Then, we choose one of the DataFrames for further processing.
#
# Next, we define `out_folder` as the folder where the plots will be saved.
# `params_path` is the file where the fitting parameters will be saved.

# %%
# load data
cfg = dl.load_config("config.yml")
data = dl.load_data_hdf(cfg["datafile"])

# chose dataset
name = "BDN2"
df = data[name]

# output folder
out_folder = Path(cfg["output_folder"]) / f"ball_predictions/{name}/"

# parameter file
params_path = out_folder / "fit_params.yml"

# output file with ball prediction columns added
output_file = out_folder / "df_ballpredictn.parquet"

# %% [markdown]
# The `refine_fit_wrapper` function will cycle through all flies in `df`
# and fit the ball with the default parameter set defined in the `cfg` file.
# For each fly it will create a subfolder in `out_folder` and the following plots:
# - `r_distr_trial_{tnum}.png`
# - `stepcycles_trial_{tnum}.png`
# See `ball_fitting_example.py` for an explanation of the plots and parameters.
#
# Furthermore, the `params_path` file will be created after the first run,
# containing the fitting parameters used for each fly.
# The `params_path` file contains 'old' and a 'new' parameter set.
# If you modify the 'new' parameter set, the fitting will be rerun.
#
# You can now:
# - modify the 'global' parameter set will apply the new parameters to all flies
# - modify either of the 'fly_{flynum}' parameter sets will only affect that fly
#
# Running `refine_fit_wrapper` again will look for changes in the `params_path` file
# and rerun the analysis for the flies that have been modified.

# %%
# (re)run fitting
bh.refine_fit_wrapper(df, out_folder, params_path, cfg["ball_fitting_defaults"])

# %% [markdown]
# Once you are happy with the results, you can run the `add_stepcycle` function.
# This will read the latest parameters from `params_path` and add the
# stepcycle predictions as well as the distances of all points from the ball center to the `df` DataFrame.
#
# Finally, we write the updated DataFrame to the original data file (or write to a new file if you prefer).

# %%
# add stepcycle predictions to df based on refined parameters
bh.add_stepcyles(df, params_path)

# add back to data dict
data[name] = df

# save to disk
dl.write_data_dict(data, output_file)

# %% [markdown]
# # generate XYZ files for all fits

# %%
# generate files for all data
for gen, df_gen in data.items():
    print(f"INFO generating xyz for {gen}")
    for flynum, df_fly in df_gen.groupby("flynum"):
        ball = [
            df.loc[:, "x_ball"].unique().item(),
            df.loc[:, "y_ball"].unique().item(),
            df.loc[:, "z_ball"].unique().item(),
        ], 
        r = df.loc[:, "r_ball"].unique().item()
        xyz.write_xyz(df_fly, out_name=f"{gen}_fly{flynum}.xyz", split=1400, ball=ball)
        print(f'INFO gen {gen} | fly {flynum} | r_ball {r}')
