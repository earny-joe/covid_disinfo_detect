import pickle
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from google.cloud import storage
from dask.distributed import Client
import dask.dataframe as dd
import joblib
import settings.config as config


def setup_dask_client():
    """
    Sets up Dask distributed client
    """
    client = Client()
    return client


def generate_svd_model(filepath=config.ALL_DATA_FILEPATH):
    """
    Generates an SVD model based on a sample of from all available
    day's of COVID-19 Twitter data
    """
    # load in Dask dataframe
    df = dd.read_csv(
        filepath,
        dtype={'tweet_id': str}
    ).sample(frac=0.01, random_state=8).reset_index(drop=True)

    # Dask array of embedding values for all observations
    X = df.iloc[:, 1:].to_dask_array()

    # init Truncated SVD object
    svd = TruncatedSVD(n_components=2, random_state=8)

    # use Dask to fit TruncatedSVD model to embedding samples
    with joblib.parallel_backend('dask'):
        svd.fit(X)

    # save pickle file of SVD model generated
    pickle.dump(
        svd,
        open(f'{config.MODEL_DIRECT_PATH}/{config.MODEL_FILE_NAME}', 'wb')
    )

    # return SVD model for further use
    return svd


def model_file_check():
    """
    Checks to see if the file for a given model exists.
    """
    model_path = config.MODEL_DIRECT_PATH / config.MODEL_FILE_NAME
    model_exists = model_path.is_file()

    if model_exists:
        print('SVD model detected, loading pickle file.\n')
        model = pickle.load(open(f'{model_path}', 'rb'))
        return model
    else:
        print('No SVD model detected, generating TruncatedSVD model.\n')
        model = generate_svd_model()
        return model


def list_csv_embedding_dates(bucket_name=config.BUCKET_NAME):
    '''
    Gathers the dates of embedding-related CSV files already stored in GCS
    '''
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')
    csv_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_embeddings.csv')
    ]
    csv_embed_dates = [
        i.split('/')[1] for i in csv_files
    ]
    return csv_embed_dates


def list_csv_dr_embed_dates(bucket_name=config.BUCKET_NAME):
    '''
    Gathers the dates of embedding-related parquet files already stored in GCS
    '''
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')
    parquet_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_embeddings_svddr.parquet')
    ]
    parquet_dr_embed_dates = [
        i.split('/')[1] for i in parquet_files
    ]
    return parquet_dr_embed_dates


def dates_need_dimen_reduce():
    '''
    Gather dates that have parquet files, but no accompanying embedding
    parquet file.
    '''
    embedding_dates = list_csv_embedding_dates()
    dr_embed_dates = list_csv_dr_embed_dates()
    need_dr_embeds = sorted(list(
        set(embedding_dates) - set(dr_embed_dates)
    ))
    # for testing purposes only retrieving two days
    return need_dr_embeds[-2:][::-1]


def clean_parquet_data(df):
    """
    Cleans parquet data
    """
    # subset of English-only tweets
    df = df[df['lang'] == 'en'].reset_index(drop=True)
    # gather lengths of tweets
    df['tweet_length'] = df['full_text'].apply(lambda x: len(x))
    # remove tweets that are abnormally long
    df = df[df['tweet_length'] <= 330].reset_index(drop=True)
    # then drop column (we don't need it any further)
    df = df.drop(columns='tweet_length')

    return df


def load_parquet_data(day):
    '''
    Takes path to tweet parquet file in GCS bucket and
    returns a Pandas DataFrame
    '''
    bucket_path_parquet = (
        f'gs://my_sm_project_data/dailies/{day}/{day}_twitter_data.parquet'
    )
    # load in parquet file
    df = pd.read_parquet(
        bucket_path_parquet
    )
    # clean up parquet data
    df = clean_parquet_data(df)

    return df


def load_embedding_csv_data(bucket_path):
    """
    Given path to CSV in GCS, loads csv into a Dask DataFrame
    """
    df = dd.read_csv(
        bucket_path,
        dtype={'tweet_id': str}
    )
    return df


def gather_tweet_ids(df):
    """
    Given a Dask Dataframe, retrieves and stores in memory the Tweet IDs
    for the respective day's embeddings, which will later be joined to the
    2-dimensional embeddings
    """
    tweet_ids = df['tweet_id'].to_dask_array()
    return tweet_ids


def gather_daily_embedding_data(day):
    """
    Gathers both embedding data for calculation, and loads in metadata associated
    with day's tweets to be combined in final data set.
    """
    bucket_path_csv = (
        f'gs://my_sm_project_data/dailies/{day}/{day}_embeddings.csv'
    )
    # gather dataframe for embeddings
    df_embeddings = load_embedding_csv_data(bucket_path_csv)
    # gather tweet IDs
    tweet_ids = gather_tweet_ids(df_embeddings)
    # gather values of the embeddings
    X = df_embeddings.iloc[:, 1:].to_dask_array()

    return tweet_ids, X


def perform_svd_embeddings(svd, X):
    """
    Given SVD model and Dask array containing embedding values, apply
    TruncatedSVD to reduce dimensionality to 2.
    """
    with joblib.parallel_backend('dask'):
        X_svd = svd.transform(X)

    return X_svd


def output_data_set(input_tweet_ids, X_svd, df_tweets):
    """
    Given two numpy arrays and a dataframe, combine into final
    data reduced dimensionality data set for visualization.
    """
    df = pd.DataFrame(X_svd, columns=('x', 'y'))
    df['tweet_id'] = input_tweet_ids
    df['created_at'] = df_tweets['created_at']
    df['tweet'] = df_tweets['full_text']

    return df[['tweet_id', 'created_at', 'tweet', 'x', 'y']]


def upload_dr_data_gcs(df_final, day):
    """
    Converts pandas DataFrame for a given day's DR embeddings into
    parquet file for storage
    """
    df_final.to_parquet(
        f'gs://my_sm_project_data/dailies/{day}/{day}_embeddings_svddr.parquet'
    )
    print(f'Dataframe of DR embeddings for {day} uploaded to bucket as parquet file.\n')


def main():
    model = model_file_check()
    need_dr_embeds = dates_need_dimen_reduce()

    for day in need_dr_embeds:
        print(f'Transforming {day} data...\n')
        df_tweets = load_parquet_data(day)
        tweet_ids, X = gather_daily_embedding_data(day)
        X_svd = perform_svd_embeddings(model, X)
        input_tweet_ids = tweet_ids.compute()
        df_final = output_data_set(input_tweet_ids, X_svd, df_tweets)
        upload_dr_data_gcs(df_final, day)


if __name__ == "__main__":
    client = setup_dask_client()
    print(f'Dask Client information: {client}\n')
    main()
