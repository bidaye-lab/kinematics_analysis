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
# %load_ext autoreload
# %autoreload 2

from src import (
    data_loader as dl,
    df_operations as dfo,
    fitting as fit,
    visualize as vis,
)

# %% [markdown]
# # data structure
# ## load data
# The data we work with is x, y, and z coordinates for each leg joint of the
# fruit fly at each frame of the video.
#
# The data is stored as an HDF file and may contain multiple datasets, here, different genotypes.
# The output of the `load_data_hdf` function is a dictionary mapping the
# genotype name to the pandas.DataFrame for that genotype.

# %%
# load HDF file as dict of dataframes
cfg = dl.load_config('config.yml')
data = dl.load_data_hdf(cfg['datafile'])

# %% [markdown]
# ## choose fly and trial
# This example shows how to choose some genotype and fly from the data.
# We then use `filter_frames`, which by default selects all frames with optical stimulation.
# Also, any previous stepcyle predictions are removed.

# %%
# load specific fly
df = data['P9RT'].groupby('flynum').get_group(7)

# select only stim frames
df = dfo.filter_frames(df)

# drop previous stepcycle predictions
df = dfo.remove_stepcycle_predictions(df)

# %% [markdown]
# # Stepcycle predictions
# To get stepcycle predictions, we use the x, y, and z coordinates of the tarsal tips of all legs.
# The column triplet `R-F-TaG_{xyz}` indicates the x, y, and z coordinates of the tarsal tip of the right front leg, for example.
#
# The following steps are performed:
# 1. fitting of the ball center and radius
# 2. extraction of step cycles, i.e., in which frames are the legs in stance and swing
#
# ## Fitting the ball
# The function `fit_ball` fits a sphere to the positions of the tarsal tips.
#
# The first step is to get an initial guess for the ball center and radius.
# The function `get_ball0` returns the initial guess for the ball center, 
# which is defined as `s_ball0 * dWH * r_notum-TaG`, where `s_ball0` is a scaling factor
# passed to `fit_ball`, `dWH` is the distance between the wing hinges, and `r_notum-TaG` is the
# vector from the notum to the position of the tarsal tip averaged over all legs.
#
# The initial guess for the ball radius is `s_r0 * dWH`, where `s_r0` is a scaling factor
# passed to `fit_ball` and `dWH` is distance between the wing hinges.
#
# Then, the ball center and radius are optimized by minimizing the function `cost_fun_ball`.
# Here, the cost function is the sum of squared distances between the ball surface and the tarsal tips.
# In each iteration of the optimization, only the points within the xth and yth percentile of the distance
# to the ball center are used.
# The lower bound excludes possible tracking errors, where the tarsal tips would be inside the ball.
# The upper bound excluded the tarsal tips of the legs that are in swing.
# The percentile ranges are defined per leg in the dictionary `pct_range`.
#
# The percentile ranges have to be adjusted for each dataset and each leg:
# If the tarsal tips suffer from many tracking errors, the lower bound should be increased.
# If the data sets include many frames with legs in swing, the upper bound should be decreased.
#
# You can also select a subset of the data with good tracking or only stance phases to fit the ball
# and use this fit to extract step cycles from the full dataset.

# %%
# define percentiles for each leg
pct_range = {
    'R-F':  (5, 85),
    'R-M':  (5, 85),
    'R-H':  (5, 85),
    'L-F':  (5, 85),
    'L-M':  (5, 85),
    'L-H':  (5, 85),
}
# fit ball
ball, r = fit.fit_ball(df, pct_range)
print('INFO: optimized ball center x = {:1.3f} y = {:1.3f} z = {:1.3f} | radius {:1.3f}'.format(*ball, r))

# add distances from center to df
df = dfo.add_distance(df, ball)

# %% [markdown]
# The visualization of the tarsal tip distances from the ball cetner are an important quality control.
# The percentile ranges should be adjusted that the center distribution is narrow and centered around the ball radius.
#
# The left distribution shows the tarsal tips within the ball.
# The right distribution shows the tarsal tips further from the surface, e.g., while the legs are in swing phase.

# %%
# visualize TaG distribution incl percentiles
vis.plot_r_distr(df, 'TaG_r', pct_range)

# %% [markdown]
# ## Stepcyle predictions
# The stepcycle predictions take the results from the ball fitting to predict if a leg is in stance or swing.
#
# The fuction `get_r_median` returns the mean of the center distribution shown in `plot_r_distr`.
#
# Depending on the distance defined in `d_delta_r` from this mean,
# the function `add_stepcycle_pred` classifies that point as stance or swing.
# Furthermore, the parameters `min_on` and `min_off` define the minimum number of frames
# for a stance or swing phase, respectively.
# This is done in order to avoid an interruption of each phase through the misclassification of just a some frames.

# %%
# get "median" for TaG_r for each leg
d_med = dfo.get_r_median(df, pct_range)

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
df = dfo.add_stepcycle_pred(df, d_med, d_delta_r, min_on, min_off)

# %% [markdown]
# Visualizing the stepcycle predictions is an important quality control to check if the
# thresholds in `d_delta_r`, `min_on`, and `min_off` are set correctly.
#
# The values in `d_med` are solid horizontal lines, those in `d_delta_r` are dashed horizontal lines.
# Frames classified as stance are shown in blue, those classified as swing are shown in orange.

# %%
# plot example trial
df_trl = df.groupby('trial').get_group(1)
vis.plot_stepcycle_pred(df_trl, d_med, d_delta_r)
