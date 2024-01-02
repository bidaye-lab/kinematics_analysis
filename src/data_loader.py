from pathlib import Path
import yaml
import pandas as pd

from src.df_operations import unify_columns

def load_config(config):
    """Load config yml as dict.

    Parameters
    ----------
    config : str
        Path to config yml file

    Returns
    -------
    cfg : dict
        dictionary
    """

    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


def load_data_hdf(datafile):
    """Load data from location defined in datafile

    Parameters
    ----------
    datafile : path-like
        location of data file

    Returns
    -------
    data : dict
        Dictionary with genotypes as keys and pd.DataFrames with coordinates as values
    """


    print("INFO: loading file {}".format(datafile))
    print()
    dfs = pd.read_hdf(datafile)

    data = dict()
    for k, d in dfs.groupby("Genotype"):
        print("INFO: found genotype {}".format(k))

        df = d.loc[:, "flydata"].item()
        df = unify_columns(df)

        n_fly = df.loc[:, "flynum"].nunique()
        print(f"INFO: found {n_fly} flies")
        print(f"      fly (trials): ", end="")
        for fly, df_fly in df.groupby("flynum"):
            n_trial = df_fly.loc[:, "tnum"].nunique()
            print(f"{fly} ({n_trial}) ", end="")
        print()
  
        data[k] = df

        print()

    return data


def write_data_dict(data, path):
    """Store dict of DataFrames as single parquet file in folder.

    Parameters
    ----------
    data : dict
        Keys are genotypes, values are dataframes
    path : path-like
        Path to store data
    """

    l = []
    for gen, df in data.items():
        df.loc[:, "genotype"] = gen
        l.append(df)

    df_tot = pd.concat(l)

    print("INFO: writing file {}".format(path))

    df_tot.to_parquet(path)


# def load_data_dict(path):
#     """Load parquet file as dict of dataframes

#     Parameters
#     ----------

#     path : path-like
#         Path to parquet files

#     Returns
#     -------
#     data : dict
#         Dict containing dataframe per genotype
#     """

#     print("INFO: loading file {}".format(path))

#     df_tot = pd.read_parquet(path)

#     data = dict()
#     # cycle through genotype
#     for gen, df in df_tot.groupby("genotype"):
#         data[gen] = df

#     return data


# def load_ball_centers(cfg):
#     """Load ball centers based on CSV files defined in config

#     Parameters
#     ----------
#     cfg : dict
#         config dict with paths to files


#     Returns
#     -------
#     ball_centers : dict
#         Mapping between genotype name and ball center (3d np.array)
#     """

#     ball_centers = dict()
#     for k, v in cfg.items():
#         if k.endswith("_ball"):
#             path = Path(v)
#             df = pd.read_csv(path, index_col=0)
#             xyz = df.iloc[0, :].values
#             gen = k.split("_")[0]
#             print(
#                 "INFO: setting ball center for {} from file {}".format(gen, path.name)
#             )
#             print("      x = {:1.3f}, y = {:1.3f}, z = {:1.3f}".format(*xyz))
#             ball_centers[gen] = xyz

#     return ball_centers
