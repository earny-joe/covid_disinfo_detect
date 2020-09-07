'''
This script is a work-in-progress and needs further development.
It is attempting to save the embeddings as numpy files, and Tweet IDs as
a CSV, as opposed to both being in a single CSV, which results in huge file
size.
'''
import utils
import pandas as pd
from google.cloud import storage
from tqdm.auto import tqdm
from nltk.tokenize import TweetTokenizer
from emoji import demojize
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from pprint import pprint
from tensorflow.python.lib.io import file_io
from settings.config import EMBED_MODEL_NAME, BUCKET_NAME
tqdm.pandas()


def list_parquet_tweet_dates(bucket_name=BUCKET_NAME):
    '''
    Gathers the dates of tweet-related parquet files already stored in GCS
    '''
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')
    parquet_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_twitter_data.parquet')
    ]
    parquet_tweet_dates = [
        i.split('/')[1] for i in parquet_files
    ]
    return parquet_tweet_dates


def list_npy_embedding_dates(bucket_name=BUCKET_NAME):
    '''
    Gathers the dates of embedding-related parquet files already stored in GCS
    MODIFICATION OF list_csv_embedding_dates function in orig file
    '''
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')
    npy_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_embeddings.npy')
    ]
    npy_embed_dates = [
        i.split('/')[1] for i in npy_files
    ]
    return npy_embed_dates


def dates_need_embeddings():
    '''
    Gather dates that have parquet files, but no accompanying embedding
    parquet file.
    '''
    parquet_tweet_dates = list_parquet_tweet_dates()
    csv_embed_dates = list_npy_embedding_dates()

    need_embeddings = sorted(list(
        set(parquet_tweet_dates) - set(csv_embed_dates)
    ))

    # for testing purposes only grabbing two most recent dates
    return need_embeddings[::-1][:2]


def normalize_token(token):
    lwrcase_tok = token.lower()
    if token.startswith('@'):
        return '@USER'
    elif lwrcase_tok.startswith('http') or lwrcase_tok.startswith('www'):
        return 'HTTPURL'
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalize_tweet(tweet):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalize_token(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    normTweet = re.sub(
        r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet
    )
    normTweet = re.sub(
        r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet
    )
    normTweet = re.sub(
        r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet
    )

    return " ".join(normTweet.split())


def clean_for_embeddings(df):
    '''
    Given input dataframe, creates subset of only English
    tweets, and applies text normalization according to
    normalize_tweet function above.
    '''
    print('Cleaning data & applying text normalization...\n')
    # subset of English-only tweets
    df_english = df[df['lang'] == 'en'].reset_index(drop=True)

    # text normalization
    df_english['normalized_tweet'] = df_english['full_text'].progress_apply(
        lambda tweet: normalize_tweet(tweet)
    ).str.lower()

    return df_english


def load_parquet_data(bucket_path):
    '''
    Takes path to tweet parquet file in GCS bucket and
    returns a Pandas DataFrame
    '''
    # load in parquet file
    df = pd.read_parquet(
        bucket_path
    )
    # gather lengths of tweets
    df['tweet_length'] = df['full_text'].apply(lambda x: len(x))
    # remove tweets that are abnormally long
    df = df[df['tweet_length'] <= 330].reset_index(drop=True)
    # then drop column (we don't need it any further)
    df = df.drop(columns='tweet_length')

    return df


def embedding_data_prep_wrapper(day):
    """
    Preps data for embeddings
    """
    bucket_path = (
        f'gs://my_sm_project_data/dailies/{day}/{day}_twitter_data.parquet'
    )
    # load tweet parquet data
    df = load_parquet_data(bucket_path)
    # gather English tweets and text normalization
    df = clean_for_embeddings(df)

    return df


def create_embedding_model(embed_model_name):
    '''
    Given string of pretrain embedding available in sentence-transformers
    library, create a SentenceTransformer object to encode embeddings with
    '''
    print('Loading SentenceTransformer...')
    model = SentenceTransformer(embed_model_name)
    print(f'{embed_model_name} model loaded!\n')
    return model


def generate_embeddings(model, tweets, day):
    '''
    Given a SentenceTransformer model object, and a tweets object,
    containing a pandas Series of tweets, use embedding model to
    encode tweets with model object.
    '''
    print('Generating tweet embeddings...\n')

    # encode the tweets
    tweet_embeddings = model.encode(
        tweets,
        batch_size=4,
        show_progress_bar=True,
        num_workers=8
    )
    # since embeddings returned as list of arrays, convert to numpy array
    return np.array(tweet_embeddings)


def generate_embedding_df(tweet_ids, tweet_embeddings):
    '''
    Given a series of tweet IDs and a list of tweet_embeddings
    (where each observation in the list is an array containing the
    embeddings), combines the two to produce a pandas DataFrame
    '''
    print('Generating pandas DataFrame with IDs and embeddings...\n')
    df = pd.DataFrame(tweet_embeddings)

    # apply more appropriate column names
    old_cols = df.columns.values
    new_cols = ['embed_' + str(x + 1) for x in old_cols]
    df.columns = new_cols

    # insert tweet IDs series as first column
    df.insert(loc=0, column='tweet_id', value=tweet_ids)

    return df


def save_embeddings_npy_file(tweet_embeddings, day):
    """
    123
    """
    dest = f'gs://my_sm_project_data/dailies/{day}/{day}_embeddings.npy'

    f = file_io.FileIO(dest, 'w')
    np.save(
        f, tweet_embeddings
    )

    print(f'\nEmbeddings for {day} saved at: {dest}\n')


def main():
    # set seeds
    utils.set_seed()

    # gather dates that need embeddings
    need_embeddings = dates_need_embeddings()
    print('Dates that need embeddings')
    print('-' * 30)
    pprint(need_embeddings)
    print('\n')

    # create Sentence Transformer model from distilbert
    embed_model_name = EMBED_MODEL_NAME
    model = create_embedding_model(embed_model_name)

    for day in need_embeddings:
        print(f'Generating embeddings for {day}.\n')
        # gather data
        df = embedding_data_prep_wrapper(day)
        # gather tweets
        tweets = df['normalized_tweet'].tolist()

        # save tweet IDs as CSV
        print('\nSaving tweet IDs as CSV file...\n')
        csv_file_path = f'gs://my_sm_project_data/dailies/{day}/{day}_embeddings_ID.csv'
        df['id_str'].to_csv(
            csv_file_path,
            index=False
        )
        print(f'Tweet IDs for {day} saved at: {csv_file_path}\n')

        # gen embeddings as store as NPY file in GCS
        tweet_embeddings = generate_embeddings(model, tweets, day)
        # save as NPY file
        save_embeddings_npy_file(tweet_embeddings, day)


if __name__ == '__main__':
    main()
