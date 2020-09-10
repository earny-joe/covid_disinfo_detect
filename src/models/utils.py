import os
import random
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re
from settings.config import SEED_VALUE, EMBED_MODEL_NAME, BUCKET_NAME
from tqdm.auto import tqdm
from google.cloud import storage
tqdm.pandas()


def set_seed(seed=SEED_VALUE):
    """
    Set the random seed for generating embeddings
    """
    # Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)
    # set `torch` pseudo-random generator at a fixed value
    torch.manual_seed(seed)
    print(f'\nSeed value set as {seed}.\n')


def list_parquet_tweet_files(bucket_name=BUCKET_NAME):
    """
    Gathers the dates of tweet-related parquet files already stored in GCS
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')
    parquet_files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_twitter_data.parquet')
    ]
    return parquet_files


def list_parquet_tweet_dates(bucket_name=BUCKET_NAME):
    """
    Gathers the dates of tweet-related parquet files already stored in GCS
    """
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


def create_embedding_model(embed_model_name=EMBED_MODEL_NAME):
    '''
    Given string of pretrain embedding available in sentence-transformers
    library, create a SentenceTransformer object to encode embeddings with
    '''
    print('Loading SentenceTransformer...')
    model = SentenceTransformer(embed_model_name)
    print(f'{embed_model_name} model loaded!\n')
    return model


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
