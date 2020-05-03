import pandas as pd
from pathlib import Path


def load_data(filename):
    """
    Given path to a specific data directory, loads in data from given filename
    """
    # change directory to where data is located
    datapath = Path.cwd() / "playground_data"
    # load in data with given filename
    df = pd.read_pickle(datapath/filename)
    # return dataframe
    return df
