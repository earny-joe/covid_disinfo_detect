"""
Beginning work towards script that'll fine-tune model for tweets
"""
from fastai.text.all import *
from transformers import AutoModelForMaskedLM, AutoTokenizer
import utils
import pandas as pd
from sklearn.model_selection import train_test_split
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


def split_data_train_test(df):
    """
    Given pandas Dataframe of tweets, splits into a train and test set
    for fine-tuning assessment.
    """
    train, test = train_test_split(
        df['normalized_tweet'],
        test_size=0.2,
        random_state=config.SEED_VALUE,
        shuffle=True
    )
    return train, test


class TransformersTokenizer(Transform):
    """
    Tokenizer class for fine-tuning (from fast.ai docs)
    """
    def __init__(self, tokenizer):
        # init
        self.tokenizer = tokenizer

    def encodes(self, x):
        # encodes tokens
        toks = self.tokenizer.tokenize(x, max_length=512)
        return tensor(self.tokenizer.encode(toks))

    def decodes(self, x):
        # decodes tokens
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


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

    train, test = split_data_train_test(df)
    print(len(train), len(test))

    splits = [range_of(train), list(range(len(train), len(all_texts)))]
    tls = TfmdLists(
        all_texts,
        TransformersTokenizer(tokenizer),
        splits=splits,
        dl_type=LMDataLoader
    )
    print(tls.train[0], '\n', tls.valid[0], '\n')
    bs, sl = 8, 512
    dls = tls.dataloaders(bs=bs, seq_len=sl)
    dls.show_batch(max_n=5)

    # num = random.randint(0, len(df))
    # test_tweet = all_texts[num]
    # print(test_tweet, '\n')
    # test_ids = tokenizer.encode(test_tweet)
    # print(test_ids, '\n')
    # print(tokenizer.decode(test_ids))

    # norm_tweets = num_tokens_avg(df)
    # print(norm_tweets.mean(), norm_tweets.std(), norm_tweets.min(), norm_tweets.max())


if __name__ == "__main__":
    main()
