import os
import pandas as pd


def load_data(origpath, datapath, filename):
    """
    Given path to a specific data directory, loads in data from given filename
    """
    # change directory to where data is located
    os.chdir(datapath)
    # load in data with given filename
    df = pd.read_pickle(filename)
    # change directory back to original
    os.chdir(origpath)
    # return dataframe
    return df
