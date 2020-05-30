import pandas as pd


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
    Takes Panda DataFrame, cleans data into appropriate form for BigQuery.
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
        f'gs://thepanacealab_covid19twitter/dailies/{day}/{day}_tweets.parquet'
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


def main():
    print('Date Format:\tYYYY-MM-DD')
    day = input('What day would you like to convert to parquet?\n')
    data_prep_wrapper(day)


if __name__ == '__main__':
    main()
