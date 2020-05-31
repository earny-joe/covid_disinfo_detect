import pandas as pd
from google.cloud import storage


def load_from_gcs(bucket_path):
    '''
    Takes path to file in Google Cloud Storage Bucket
    and returns a Pandas DataFrame
    '''
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
    '''
    Takes Panda DataFrame, cleans data into appropriate
    form for BigQuery.
    '''
    cols_of_interest = [
        'created_at',
        'id_str',
        'user',
        'lang',
        'full_text'
    ]
    df2clean = df.loc[:, cols_of_interest]
    df2clean['user_id_str'] = df2clean['user'].apply(
        lambda user: str(user['id_str'])
    )
    df2clean.drop(labels='user', axis=1, inplace=True)
    dfclean = df2clean[
        ['created_at',
         'id_str',
         'user_id_str',
         'lang',
         'full_text']
    ]
    return dfclean


def parquet_to_gcs(df, day):
    df.to_parquet(
        f'gs://thepanacealab_covid19twitter/dailies/'
        + f'{day}/{day}_tweets.parquet'
    )
    print('Dataframe uploaded to bucket as parquet file.')


def data_prep_wrapper(day):
    bucket_path = (
        f'gs://thepanacealab_covid19twitter/dailies/'
        + f'{day}/{day}_clean-dataset.json'
    )
    print(f'Loading data for {day}...')
    df = load_from_gcs(bucket_path)
    print(f'Cleaning data for {day}...')
    dfclean = clean_for_parquet(df)
    assert len(dfclean) == len(df)
    print(f'Converting to parquet file & storing in {day} bucket.')
    parquet_to_gcs(dfclean, day)
    print(f'{day} successfully converted and stored in Storage.')


def list_json_dates(bucket_name='thepanacealab_covid19twitter'):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')
    json_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('.json')
    ]
    json_dates = [i.split('/')[1] for i in json_files]
    return json_dates


def list_parquet_dates(bucket_name='thepanacealab_covid19twitter'):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')
    parquet_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('.parquet')
    ]
    parquet_dates = [
        i.split('/')[1] for i in parquet_files
    ]
    return parquet_dates


def need_parquet_dates():
    json_dates = list_json_dates()
    parquet_dates = list_parquet_dates()
    need_parquet = sorted(list(set(json_dates) - set(parquet_dates)))
    return need_parquet


def main():
    need_parquet = need_parquet_dates()
    print(
        f'Need to generate {len(need_parquet)} for the following dates:\n'
        + f'{need_parquet}\n'
    )
    for day in need_parquet:
        print(f'Converting data for {day}...')
        data_prep_wrapper(day)


if __name__ == '__main__':
    main()
