import pandas as pd
from pathlib import Path
from google.cloud import storage
import re


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


def entity_extraction(entity, component, urls=False, user_mentions=False):
    try:
        if urls is True:
            if entity[component] == []:
                return None
            elif entity[component] != []:
                return ','.join([url['url'] for url in entity[component]])
        elif user_mentions is True:
            if entity[component] == []:
                return None
            elif entity[component] != []:
                return ','.join(
                    [mention['screen_name'] for
                     mention in entity[component]]
                )
        else:
            if entity[component] == []:
                return None
            elif entity[component] != []:
                return ','.join([comp['text'] for comp in entity[component]])
    except Exception:
        return None


def source_extract(text):
    try:
        regex = re.compile(r'(?<=>).*?(?=<)', re.I)
        return regex.search(text).group()
    except AttributeError:
        return None


def quoted_status_extract(status):
    try:
        return status['url']
    except Exception:
        return None


def clean_panacea_data(dataframe):
    user_components = [
        'created_at',
        'description',
        'favourites_count',
        'followers_count',
        'friends_count',
        'id_str',
        'location',
        'name',
        'profile_image_url_https',
        'screen_name',
        'statuses_count',
        'verified'
    ]
    dataframe['hashtags'] = dataframe['entities']\
        .apply(lambda x: entity_extraction(x, 'hashtags'))
    dataframe['symbols'] = dataframe['entities']\
        .apply(lambda x: entity_extraction(x, 'symbols'))
    dataframe['urls'] = dataframe['entities']\
        .apply(lambda x: entity_extraction(x, 'urls', urls=True))
    dataframe['user_mentions'] = dataframe['entities']\
        .apply(lambda x: entity_extraction(x, 'user_mentions',
                                           user_mentions=True))
    dataframe['tweet_source'] = dataframe['source'].apply(source_extract)
    for comp in user_components:
        dataframe[f'user_{comp}'] = dataframe['user']\
            .apply(lambda user: user[comp])
    dataframe['quoted_status_url'] = dataframe['quoted_status_permalink']\
        .apply(quoted_status_extract)
    dataframe.drop(
        labels=[
            'user',
            'entities',
            'source',
            'quoted_status_permalink'
        ], axis=1, inplace=True
    )
    dataframe.fillna('none', inplace=True)
    return dataframe
