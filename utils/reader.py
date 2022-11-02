import pandas as pd


def read_data_from_file(path_to_file):
    return pd.read_csv(path_to_file, sep=':', index_col=0, names=["User", "Data"])
