import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


def save(fig, path=""):
    """Save figure to path and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save and close.
    path : path-like, optional
        Path to save figure to. If emtpy string, figure is not saved.
    """

    if path:
        fig.savefig(path)
        plt.close(fig)


def plot_r_distr(df, col_match, pct_range={}, xlims=(None, None), path=""):
    """Plot distribution of distances from ball center.

    This is a quality control for the surface fitting.
    Each leg is plotted in a separate subplot.

    Parameters
    ----------
    df : pandas.DataFrame
        Distance data.
    col_match : str
        Plot columns containing this string.
    pct_range : dict, optional
        Percentiles per leg used in fitting, by default {}
    xlims : tuple, optional
        Manually set x axis limits, by default (None, None)
    path : path-like, optional
        If not '', save plot to file, by default ''
    """
    # construct xyz str based on joint
    cols = [c for c in df.columns if col_match in c]

    fig, axmat = plt.subplots(
        nrows=len(cols) // 2, ncols=2, figsize=(10, len(cols) * 1.5)
    )

    rmin = df.loc[:, cols].min().min()
    rmax = df.loc[:, cols].max().max()

    for ax, c in zip(axmat.T.flatten(), cols):
        r = df.loc[:, c].values
        perc = pct_range.get(c[:3], [5, 95])

        # create arrays for three intervals
        a, b = np.nanpercentile(r, perc)
        r1 = r[r < a]
        r2 = r[(r > a) & (r < b)]
        r3 = r[r > b]

        # plot
        sns.histplot(
            data=[r1, r2, r3],
            ax=ax,
            legend=False,
            binrange=(rmin, rmax),
            binwidth=0.02,
            multiple="stack",
        )
        ax.set_title(c)
        ax.set_xlim(xlims)

    fig.tight_layout()
    save(fig, path)


def plot_stepcycle_pred(df, d_med, d_delta_r, vspan=(), path=""):
    """Plot stepcycle predictions indicating stance and swing phases.

    This plots the `TaG_r` values for each leg vs time and colors
    the points according to stance (blue) and swing (orange) phases.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing `{leg}_stepcycle` columns.
    d_med : dict
        Surface location for each leg.
    d_delta_r : dict
        Distance from surface threshold for each leg.
    vspan : tuple, optional
        If not empty, plot a vertical span from `vspan[0]` to `vspan[1]` in the background, by default ()
    path : str, optional
        If not '', save plot to file, by default ''
    """

    cols = [c for c in df.columns if "TaG_r" in c]
    fig, axarr = plt.subplots(nrows=len(cols), figsize=(20, 20))

    for ax, (leg, delta_r) in zip(axarr, d_delta_r.items()):
        # on/off ball predictions
        on = df.loc[:, "{}_stepcycle".format(leg)]
        off = ~on

        # distance from ball center
        col = f"{leg}-TaG_r"
        r = df.loc[:, col]

        # corresponding frame number
        f = df.loc[:, 'fnum']

        sns.scatterplot(x=f.loc[on], y=r.loc[on], ax=ax)
        sns.scatterplot(x=f.loc[off], y=r.loc[off], ax=ax)

        # plot background box
        if vspan:
            x0 = vspan[0] + f.min()
            xf = vspan[1] + f.min()
            ax.axvspan(x0, xf, color="k", alpha=0.05)

        # plot "median" and thresh
        r_m = d_med[leg]
        ax.axhline(r_m, c="gray")
        ax.axhline(r_m + delta_r, c="gray", ls="--")

    save(fig, path)


def plot_stepcycle_pred_grid(df, d_med, delta_r, path=""):
    """Same as `plot_stepcycle_pred` but with a grid of subplots for each trial.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing `{leg}_stepcycle` columns.
    d_med : dict
        Surface location for each leg.
    d_delta_r : dict
        Distance from surface threshold for each leg.
    path : str, optional
        If not '', save plot to file, by default ''
    """

    cols = [c for c in df.columns if "TaG_r" in c]
    trials = df.loc[:, "tnum"].unique()
    fig, axmat = plt.subplots(
        nrows=len(trials), ncols=len(cols), squeeze=False, figsize=(40, 4 * len(trials))
    )

    for axarr, c in zip(
        axmat.T,
        cols,
    ):
        for ax, (t, df_t) in zip(axarr, df.groupby("tnum")):
            leg = "-".join(c.split("-")[:2])

            # on/off ball predictions
            on = df_t.loc[:, "{}_stepcycle".format(leg)]
            off = ~on

            # distance from ball center
            r = df_t.loc[:, c]

            sns.scatterplot(r.loc[on], ax=ax)
            sns.scatterplot(r.loc[off], ax=ax)
            sns.lineplot(r, ax=ax, color="gray", lw=0.5)

            # plot "median" and thresh
            r_m = d_med[leg]
            ax.axhline(r_m, c="gray")
            ax.axhline(r_m + delta_r, c="gray", ls="--")

            ax.set_title("leg {} | trial {}".format(leg, t))

    fig.tight_layout()

    save(fig, path)
