import pandas as pd
from google.cloud import storage
from pprint import pprint
from settings.config import COLS_SELECT, COLS_INTEREST, BUCKET_NAME


def load_from_gcs(bucket_path):
    """
    Takes path to file in Google Cloud Storage Bucket
    and returns a Pandas DataFrame
    """
    # load in raw JSON data
    df = pd.read_json(
        bucket_path,
        lines=True,
        dtype={
            'id_str': str,
            'in_reply_to_status_id_str': str,
            'quoted_status_id_str': str
        }
    )
    return df


def clean_for_parquet(df):
    """
    Cleans up raw data, returns dataframe with subset that we're
    interested in.
    """
    # select all rows, and subset of columns
    df2clean = df.loc[:, COLS_SELECT]
    # get user ID
    df2clean['user_id_str'] = df2clean['user'].apply(
        lambda user: str(user['id_str'])
    )
    # drop user column (contains a lot of irrelevant info)
    df2clean.drop(labels='user', axis=1, inplace=True)
    # return on columns of interest
    dfclean = df2clean[COLS_INTEREST]

    return dfclean


def parquet_to_gcs(df, day):
    """
    Converts pandas dataframe for a given day into a parquet
    file for storage.
    """
    # GCS path for file location
    parquet_bucket_path = f'gs://{BUCKET_NAME}/dailies/{day}/{day}_twitter_data.parquet'

    # save dataframe as parquet
    df.to_parquet(parquet_bucket_path)

    print(f'Dataframe uploaded to: {parquet_bucket_path}.\n')


def data_prep_wrapper(day):
    """
    Wraps functions load_from_gcs, clean_for_parquet, and
    parquet_to_gcs together to perform one after the other.
    """
    bucket_path = (
        f'gs://{BUCKET_NAME}/dailies/{day}/{day}_clean-dataset.json.gz'
    )
    print(f'\nLoading data for {day}...\n')
    df = load_from_gcs(bucket_path)

    print(f'Cleaning data for {day}...\n')
    dfclean = clean_for_parquet(df)
    assert len(dfclean) == len(df)

    print(f'Converting to parquet file & storing in {day} bucket...\n')
    parquet_to_gcs(dfclean, day)
    print(f'{day} successfully stored.\n')


def list_json_dates(bucket_name=BUCKET_NAME):
    """
    Gathers the dates of JSON files that are already stored in GCS
    """
    # init GCS client, get all blobs in bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')

    # list of dates with raw JSON data
    json_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_clean-dataset.json')
    ]
    json_dates = [i.split('/')[1] for i in json_files]

    return json_dates


def list_parquet_dates(bucket_name=BUCKET_NAME):
    """
    Gathers the data of parquet files already stored in GCS
    """
    # init GCS client, get all blobs in bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')

    # list of dates with "cleaned up" raw data
    parquet_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_twitter_data.parquet')
    ]
    parquet_dates = [
        i.split('/')[1] for i in parquet_files
    ]

    return parquet_dates


def need_parquet_dates():
    """
    Gather dates that have JSON files, but no accompanying parquet file
    """
    # dates that have JSON files
    json_dates = list_json_dates()

    # dates that have parquet files
    parquet_dates = list_parquet_dates()

    # dates we need to convert
    need_parquet = sorted(list(set(json_dates) - set(parquet_dates)))

    return need_parquet


def main():
    """
    Main application that converts all JSONs whose dates are not yet in
    parquet format, and converts them, and stores them in same folder
    together with the raw JSON
    """
    need_parquet = need_parquet_dates()[::-1]
    print('Days To Convert')
    print('-' * 30)
    pprint(need_parquet)

    for day in need_parquet:
        data_prep_wrapper(day)


if __name__ == '__main__':
    main()
