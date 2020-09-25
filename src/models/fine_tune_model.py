"""
Beginning work towards script that'll fine-tune model for tweets
"""
from transformers import AutoModelForMaskedLM, AutoTokenizer
import utils
import pandas as pd
import settings.config as config
from datetime import datetime
import numpy as np
import random


def load_daily_data(bucket_path):
    """
    Given path to GCS bucket, loads a day's data according to certain
    conditions and returns a sample to be used in fine-tuning of
    embedding model.
    """
    print(f'Loading data from file: {bucket_path}\n')
    df = pd.read_parquet(bucket_path)
    # gather lengths of tweets
    df['tweet_length'] = df['full_text'].apply(lambda x: len(x))
    # remove tweets that are abnormally long
    df = df[df['tweet_length'] <= 330].reset_index(drop=True)
    # then drop column (we don't need it any further)
    df = df.drop(columns='tweet_length')

    return df.sample(frac=0.05, random_state=config.SEED_VALUE)


def load_data_wrapper(parquet_files):
    """
    Given a list of parquet files, sample from each daily file, apply
    necessary text processing, and return a dataframe that includes
    data from all available daily Twitter data.
    """
    # testing on two files
    test = parquet_files[:2]

    begin = datetime.now()
    df = pd.concat([
        load_daily_data(f'gs://{config.BUCKET_NAME}/{file}') for file in test
    ]).reset_index(drop=True)

    df = utils.clean_for_embeddings(df)

    end = datetime.now()
    print(f'\nLoading in {len(parquet_files)} files took {end - begin}.\n')

    return df


def load_transformer_model():
    """
    TBD
    """
    tokenizer = AutoTokenizer.from_pretrained(
        f'sentence-transformers/{config.EMBED_MODEL_NAME}'
    )
    model = AutoModelForMaskedLM.from_pretrained(
        f'sentence-transformers/{config.EMBED_MODEL_NAME}'
    )
    return tokenizer, model


def main():
    """
    Main application
    """
    # set seeds
    utils.set_seed()
    # load transformers model
    tokenizer, model = load_transformer_model()

    print(model, '\n\n')

    # gather list of parquet files containing data
    parquet_files = utils.list_parquet_tweet_files()[::-1]

    df = load_data_wrapper(parquet_files)
    print(df.info(), '\n')

    all_texts = np.array(df['normalized_tweet'].values)
    print(all_texts.shape, '\n')

    num = random.randint(0, len(df))
    test_tweet = all_texts[num]
    print(test_tweet, '\n')
    test_ids = tokenizer.encode(test_tweet)
    print(test_ids, '\n')
    print(tokenizer.decode(test_ids))

    # norm_tweets = num_tokens_avg(df)
    # print(norm_tweets.mean(), norm_tweets.std(), norm_tweets.min(), norm_tweets.max())


if __name__ == "__main__":
    main()
