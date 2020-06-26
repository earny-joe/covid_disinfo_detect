'''
This script is a work-in-progress and needs further development.
Right now all of the functions below have successfully run within
a notebook that is being used in tandem to develop this script.
In the next branch, I will continue to build this out, namely by
adding a config file and also by figuring out a more appropriate
means by which to store embeddings, most likely in a database
of some sort (and not within the 'data' sub-folder of the directory)
'''
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from nltk.tokenize import TweetTokenizer
from sentence_transformers import SentenceTransformer
from emoji import demojize
import re
import umap
tqdm.pandas()


def path_to_data():
    '''
    Returns path to where parquet file is stored.
    '''
    return Path.cwd().parent.parent / 'data' / 'dailies'


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


def clean_data(df):
    '''
    Given input dataframe, creates subset of only English
    tweets, and applies text normalization according to
    normalize_tweet function above.
    '''
    # subset of English-only tweets
    df_english = df[df['lang'] == 'en'].reset_index(drop=True)

    # text normalization
    df_english['normalized_tweet'] = df_english['full_text'].apply(
        lambda tweet: normalize_tweet(tweet)
    )
    return df_english


def load_parquet_data(data_path, filename):
    '''
    Loads in file according to data_path and filename,
    applies necessary edits to return only English
    Tweets, and returns pandas dataframe
    '''
    # get folder name according to filename
    folder = filename.split('_')[0]

    # load in parquet file
    df = pd.read_parquet(
        f'{data_path}/{folder}/{filename}',
    )

    # apply necessary edits
    df = clean_data(df)

    return df


def create_embedding_model(embed_model_name):
    '''
    Given string of pretrain embedding available in sentence-transformers
    library, create a SentenceTransformer object to encode embeddings with
    '''
    model = SentenceTransformer(embed_model_name)
    return model


def generate_embeddings(model, tweets):
    '''
    Given a SentenceTransformer model object, and a tweets object,
    containing a pandas Series of tweets, use embedding model to
    encode tweets with model object.
    '''
    tweet_embeddings = model.encode(tweets)
    return tweet_embeddings


def generate_embedding_df(tweet_ids, tweet_embeddings):
    '''
    Given a series of tweet IDs and a list of tweet_embeddings
    (where each observation in the list is an array containing the
    embeddings), combines the two to produce a pandas DataFrame
    '''
    df = pd.DataFrame(tweet_embeddings)

    # apply more appropriate column names
    old_cols = df.columns.values
    new_cols = ['embed_' + str(x + 1) for x in old_cols]
    df.columns = new_cols

    # insert tweet IDs series as first column
    df.insert(loc=0, column='tweet_id', value=tweet_ids)

    return df


def generate_umap_embeddings(embeddings_df):
    '''
    Given a pandas dataframe of embeddings (with first column
    representing a respective tweet ID), apply UMAP to reduce
    dimensionality to 2, to create representation of data that
    can then be visualized in 2D space.
    '''
    # gather only embedding values (i.e. drop column with IDs)
    embedding_data = embeddings_df.iloc[:, 1:].values

    # init umap object and apply to embedding data
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.01,
        spread=0.25,
        metric='cosine',
        random_state=8,
        transform_seed=8
    )
    umap_embeddings = reducer.fit_transform(
        embedding_data
    )

    # create pandas dataframe from embeddings, with IDs as first column
    umap_embeddings_df = pd.DataFrame(
        umap_embeddings, columns=('x', 'y')
    )
    umap_embeddings_df.insert(
        loc=0, column='tweet_id', value=embeddings_df['tweet_id']
    )

    return umap_embeddings_df


def save_embeddings(filename, embeddings_df, umap_embeddings_df):
    '''
    Given a filename, and two pandas dataframe, one of all the
    embeddings, and of the UMAP embeddings (reduced to 2D),
    then save the files in same location where parquet file was
    stored.
    '''
    # get folder name according to filename (i.e. the date)
    folder_date = filename.split('_')[0]
    # retrieve path to data
    data_path = path_to_data()

    # save both embedding dataframes
    embeddings_df.to_parquet(
        f'{data_path}/{folder_date}/{folder_date}_embeddings.parquet',
    )
    umap_embeddings_df.to_parquet(
        f'{data_path}/{folder_date}/{folder_date}_umap_embeds.parquet'
    )
