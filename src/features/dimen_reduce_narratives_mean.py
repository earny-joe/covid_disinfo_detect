import pickle
import pandas as pd
import settings.config as config


def load_narrative_data():
    """
    Loads in narrative data with mean embeddings for each narrative.
    """
    # get path to narrative data
    data_path = config.NARR_STORAGE_PATH / config.MEAN_NARR_FILENAME

    # load in data at the above data path
    df_narratives = pd.read_csv(
        data_path
    )

    return df_narratives


def load_dimen_reduce_model():
    """
    Checks to see if the file for a given model exists.
    """
    # path to dr_model
    model_path = config.MODEL_DIRECT_PATH / config.MODEL_FILE_NAME

    # load dr_model from file
    print(f'\nLoading Model at:\n{model_path}.\n')
    model_dr = pickle.load(open(f'{model_path}', 'rb'))

    return model_dr


def gen_output_data(narratives, X_2d):
    """
    Gens pandas dataframe of narratives with 2d embeddings
    """
    df = pd.DataFrame(X_2d, columns=('x', 'y'))
    df['narrative'] = narratives

    return df


def transform_mean_embeddings(svd, df_narratives):
    """
    Given SVD model, transforms meamn embeddings
    """
    # gather pandas Series with each observations respective narrative
    narratives = df_narratives['narrative']
    # gather embedding values and convert to np array
    X = df_narratives.iloc[:, 1:].to_numpy()
    # transform embedding values to 2d
    X_2d = svd.transform(X)
    # generate dataframe with 2d embedding values and narratives
    df_mean_2d = gen_output_data(narratives, X_2d)

    return df_mean_2d[['narrative', 'x', 'y']]


def main():
    """
    Main application, generates 2D embedding file for misinfo narratives
    """
    file_check = config.NARR_STORAGE_PATH / config.MEAN_NARR_DR_FILENAME

    if file_check.is_file():
        print(f'\nSkipping the embedding generation step because {file_check} already exists.\n')
    else:
        df_narratives = load_narrative_data()
        model_dr = load_dimen_reduce_model()
        df_mean_2d = transform_mean_embeddings(model_dr, df_narratives)

        # save dataframe with 2d mean embeddings to CSV
        df_mean_2d.to_csv(
            f'{config.NARR_STORAGE_PATH}/{config.MEAN_NARR_DR_FILENAME}',
            index=False
        )
        print(
            '\nSaved 2D mean narrative embeddings. File located at:\n',
            f'{config.NARR_STORAGE_PATH}/{config.MEAN_NARR_DR_FILENAME}',
            '\n'
        )


if __name__ == "__main__":
    main()
