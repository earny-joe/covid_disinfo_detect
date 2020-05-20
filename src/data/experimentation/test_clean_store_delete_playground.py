# Comment 


import pandas as pd
import re
from google.cloud import storage
from pathlib import Path


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
        filename, lines=True, chunksize=chunksize,
        dtype={'id_str': str, 'in_reply_to_status_id_str': str, 'quoted_status_id_str': str}
    )
    df = pd.concat(chunk for chunk in chunks)[good_columns]
    return df


def entity_extraction(entity, component, urls=False, user_mentions=False):
    try:
        if urls == True:
            if entity[component] == []:
                return None
            elif entity[component] != []:
                return ','.join([url['url'] for url in entity[component]])
        elif user_mentions == True:
            if entity[component] == []:
                return None
            elif entity[component] != []:
                return ','.join([mention['screen_name'] for mention in entity[component]])
        else:
            if entity[component] == []:
                return None
            elif entity[component] != []:
                return ','.join([comp['text'] for comp in entity[component]])
    except:
        return None


def source_extract(text):
    try:
        regex = re.compile(r'(?<=>).*?(?=<)', re.I)
        return regex.search(text).group()
    except AttributeError as e:
        return None
    
    
def quoted_status_extract(status):
    try:
        return status['url']
    except:
        return None
    
    
def clean_panacea_data(dataframe):
    user_components = [
        'created_at', 'description', 'favourites_count', 'followers_count', 'friends_count',
        'id_str', 'location', 'name', 'profile_image_url_https', 'screen_name',
        'statuses_count', 'verified'
    ]
    dataframe['hashtags'] = dataframe['entities'].apply(lambda x: entity_extraction(x, 'hashtags'))
    dataframe['symbols'] = dataframe['entities'].apply(lambda x: entity_extraction(x, 'symbols'))
    dataframe['urls'] = dataframe['entities'].apply(lambda x: entity_extraction(x, 'urls', urls=True))
    dataframe['user_mentions'] = dataframe['entities'].apply(lambda x: entity_extraction(x, 'user_mentions', user_mentions=True))
    dataframe['tweet_source'] = dataframe['source'].apply(source_extract)
    for comp in user_components:
        dataframe[f'user_{comp}'] = dataframe['user'].apply(lambda user: user[comp])
    dataframe['quoted_status_url'] = dataframe['quoted_status_permalink'].apply(quoted_status_extract)
    dataframe.drop(labels=['user', 'entities', 'source', 'quoted_status_permalink'], axis=1, inplace=True)
    dataframe.fillna('none', inplace=True)
    return dataframe


def cleaning_wrapper(date):
    print('Loading data...')
    df = load_data(f'{date}/{date}_clean-dataset.json')
    print('Cleaning data...')
    df = clean_panacea_data(dataframe=df)
    print(f'Cleaned data, converting data for date {date} to pickle format...')
    df.to_pickle(f'{date}/{date}_clean-dataset.pkl')


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
    

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    

def main():
    date = input('Date whose data will be cleaned (format: YYYY-MM-DD):\n')
    bucket_name = 'thepanacealab_covid19twitter'
    download_blob(
        bucket_name=bucket_name,
        source_blob_name=f'dailies/{date}/panacealab_{date}_clean-dataset.json',
        destination_file_name=f'{date}/{date}_clean-dataset.json'
    )
    cleaning_wrapper(date)
    upload_blob(
        bucket_name=bucket_name,
        source_file_name=f'{date}/{date}_clean-dataset.pkl',
        destination_blob_name=f'dailies/{date}/{date}_clean-dataset.pkl'
    )
    file_delete_path = Path.cwd() / date / f'{date}_clean-dataset.json'
    file_delete_path.unlink()
    print(f'{date}_clean-dataset.json removed from {date} folder.')
    
    
if __name__ == '__main__':
    main()