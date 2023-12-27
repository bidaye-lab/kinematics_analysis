import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def save(fig, path):

    if path:
        fig.savefig(path)
        plt.close(fig)

def plot_r_distr(df, col_match, d_perc={}, xlims=(None, None), path=''):

    # construct xyz str based on joint
    cols = [ c for c in df.columns if col_match in c ]
    
    fig, axmat = plt.subplots(nrows=len(cols)//2, ncols=2, figsize=(10, len(cols)*1.5))

    rmin = df.loc[:, cols].min().min()
    rmax = df.loc[:, cols].max().max()
    
    for ax, c in zip(axmat.T.flatten(), cols):

        r = df.loc[:, c].values
        perc = d_perc.get(c[:3], [5, 95])

        # create arrays for three intervals
        a, b = np.nanpercentile(r, perc)
        r1 = r[r<a]
        r2 = r[(r>a) & (r<b)]
        r3 = r[r>b]

        # plot
        sns.histplot(data=[r1, r2, r3], 
                     ax=ax, 
                     legend=False, 
                     binrange=(rmin, rmax), 
                     binwidth=0.00025, 
                     multiple='stack'
                     )
        ax.set_title(c)
        ax.set_xlim(xlims)

    fig.tight_layout()
    save(fig, path)

def plot_stepcycle_pred(df, d_med, d_delta_r, path=''):

    cols =  [ c for c in df.columns if 'TaG_r' in c ]
    fig, axarr = plt.subplots(nrows=len(cols), figsize=(20, 20))

    for ax, (leg, delta_r) in zip(axarr, d_delta_r.items()):

        # on/off ball predictions
        on = df.loc[:, '{}_stepcycle'.format(leg)]
        off = ~on

        # distance from ball center
        col = f'{leg}-TaG_r'
        r = df.loc[:, col]

        sns.scatterplot(r.loc[on], ax=ax)
        sns.scatterplot(r.loc[off], ax=ax)
        
        # plot "median" and thresh
        r_m = d_med[leg]
        ax.axhline(r_m, c='gray')
        ax.axhline(r_m + delta_r, c='gray', ls='--')
        
    save(fig, path)