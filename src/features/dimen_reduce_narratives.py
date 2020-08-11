import sys
from pathlib import Path
import pickle
import pandas as pd
sys.path.insert(0, f'{Path.cwd()}/')
import config


def load_narrative_data():
    """
    Loads in narrative data with mean embeddings for each narrative.
    """
    data_path = config.MEAN_NARR_STORAGE_PATH / config.MEAN_NARR_FILENAME

    mean_narratives = pd.read_csv(
        data_path
    )

    return mean_narratives


def load_dimen_reduce_model():
    """
    Checks to see if the file for a given model exists.
    """
    # path to dr_model
    model_path = config.MODEL_DIRECT_PATH / config.MODEL_FILE_NAME

    # load dr_model from file
    print(f'Loading Model at:\n{model_path}.\n')
    dr_model = pickle.load(open(f'{model_path}', 'rb'))

    return dr_model


def gen_output_data(narratives, X_svd):
    """
    Gens pandas dataframe of narratives with 2d embeddings
    """
    df = pd.DataFrame(X_svd, columns=('x', 'y'))
    df['narrative'] = narratives

    return df


def transform_mean_embeddings(svd, mean_narratives):
    """
    Given SVD model, transforms mean embeddings
    """
    narratives = mean_narratives['narrative']
    X = mean_narratives.iloc[:, 1:].to_numpy()
    X_svd = svd.transform(X)

    df_mean_2d = gen_output_data(narratives, X_svd)

    return df_mean_2d[['narrative', 'x', 'y']]


def main():
    """
    Main application, generates 2D embedding file for misinfo narratives
    """
    mean_narratives = load_narrative_data()
    dr_model = load_dimen_reduce_model()
    df_mean_2d = transform_mean_embeddings(dr_model, mean_narratives)

    df_mean_2d.to_csv(
        f'{config.MEAN_NARR_STORAGE_PATH}/{config.MEAN_NARR_DR_FILENAME}',
        index=False
    )
    print(
        'Saved 2D mean narrative embeddings. File located at:\n',
        f'{config.MEAN_NARR_STORAGE_PATH}/{config.MEAN_NARR_DR_FILENAME}'
    )


if __name__ == "__main__":
    main()
