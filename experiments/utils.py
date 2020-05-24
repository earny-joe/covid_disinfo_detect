import pandas as pd
from pathlib import Path
from google.cloud import storage


def load_pkl_data(filename):
    """
    Given path to a specific data directory, loads in data from given filename
    """
    # change directory to where data is located
    datapath = Path.cwd() / "playground_data"
    # load in data with given filename
    df = pd.read_pickle(datapath/filename)
    # return dataframe
    return df


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def download_json(day):
    bucket_name = 'thepanacealab_covid19twitter'
    source_blob_name = f'dailies/{day}/{day}_clean-dataset.json'
    downloadpath = (
        Path.home() / 'covid_disinfo_detect' /
        'experiments' / 'playground_data'
    )
    download_blob(
        bucket_name,
        source_blob_name,
        downloadpath/f'{day}_clean-dataset.json'
    )


def load_data(filename, chunksize=10000):
    good_columns = [
        'created_at',
        'entities',
        'favorite_count',
        'full_text',
        'id_str',
        'in_reply_to_screen_name',
        'in_reply_to_status_id_str',
        'is_quote_status',
        'lang',
        'retweet_count',
        'source',
        'user',
        'quoted_status_id_str',
        'quoted_status_permalink'
    ]
    chunks = pd.read_json(
        f'playground_data/{filename}',
        lines=True,
        chunksize=chunksize,
        dtype={
            'id_str': str,
            'in_reply_to_status_id_str': str,
            'quoted_status_id_str': str
        }
    )
    df = pd.concat(chunk for chunk in chunks)[good_columns]
    return df
