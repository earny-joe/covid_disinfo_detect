"""
Beginning work towards script that'll fine-tune model for tweets
"""
import spacy_sentence_bert
from sentence_transformers import SentencesDataset, InputExample
import utils
import pandas as pd
import settings.config as config
from datetime import datetime
import numpy as np


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
    # testing on five files
    # test = parquet_files[:5]

    begin = datetime.now()
    df = pd.concat([
        load_daily_data(f'gs://{config.BUCKET_NAME}/{file}') for file in parquet_files
    ]).reset_index(drop=True)

    df = utils.clean_for_embeddings(df)

    end = datetime.now()
    print(f'\nLoading in {len(parquet_files)} files took {end - begin}.\n')

    return df
    
    

def gather_training_examples(df, model):
    """
    Gather training examples for fine-tuning model.
    """
    # gather all tweets
    tweets = df['normalized_tweet'].tolist()
    # pass into train_examples object
    # train_samples = []
    train_examples = InputExample(texts=tweets)
    train_dataset = SentencesDataset(train_examples, model)

    return train_dataset


def main():
    """
    Main application
    """
    # set seeds
    utils.set_seed()
    # load transformers model
    model = utils.create_embedding_model()
    #print(model.max_seq_length)
    
    # gather list of parquet files containing data
    #parquet_files = utils.list_parquet_tweet_files()[::-1]

    #df = load_data_wrapper(parquet_files)
    #norm_tweets = num_tokens_avg(df)
    # print(norm_tweets.mean(), norm_tweets.std(), norm_tweets.min(), norm_tweets.max())


if __name__ == "__main__":
    main()
