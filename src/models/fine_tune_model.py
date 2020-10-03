"""
Beginning work towards script that'll fine-tune model for tweets
"""
from fastai.text.all import *
from transformers import AutoModelForMaskedLM, AutoTokenizer
import utils
import warnings
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


def gather_all_texts(df):
    """
    Given a pandas dataframe, extracts all the values from the normalized_tweet
    column, and stores in numpy array
    """
    all_texts = np.array(df['normalized_tweet'].values)
    return all_texts


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


def data_prep_for_lm_model(train, all_texts, tokenizer):
    """
    Preps data for fine-tuning model
    """
    # train-test split
    splits = [range_of(train), list(range(len(train), len(all_texts)))]

    # transformations for training
    tls = TfmdLists(
        all_texts,
        TransformersTokenizer(tokenizer),
        splits=splits,
        dl_type=LMDataLoader
    )
    print(tls.train[0], '\n', tls.valid[0], '\n')

    dls = tls.dataloaders(
        bs=config.BATCH_SIZE,
        seq_len=config.SEQ_LENGTH
    )

    dls.show_batch(max_n=2)
    print('')
    return dls


def create_learner(dls, model):
    """
    Create learner for training
    """
    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(),
        cbs=[DropOutput],
        metrics=Perplexity()
    ).to_fp16()

    return learn


def find_learning_rate(learn):
    """
    Find the best learning rate for training.
    """
    lr_min, lr_steep = learn.lr_find(
        start_lr=1e-07,
        end_lr=10,
        num_it=100,
        stop_div=True,
        show_plot=False,
        suggestions=True
    )

    return lr_min, lr_steep


class TransformersTokenizer(Transform):
    """
    Tokenizer class for fine-tuning (from fast.ai docs)
    """
    def __init__(self, tokenizer):
        # init
        self.tokenizer = tokenizer

    def encodes(self, x):
        """
        encodes tokens, returns only first 512 tokens as that is max
        sequence length for BERT model
        """
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.encode(toks, truncation=True, max_length=512))

    def decodes(self, x):
        # decodes tokens
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


class DropOutput(Callback):
    def after_pred(self):
        # callback to alter behavior of training loop
        self.learn.pred = self.pred[0]


def main():
    """
    Main application
    """
    # filter out pandas future warning about panel
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # set seeds
    utils.set_seed()
    # load transformers model
    tokenizer, model = load_transformer_model()
    # print(model, '\n\n')
    # gather list of parquet files containing data
    parquet_files = utils.list_parquet_tweet_files()[::-1]
    # load pandas dataframe with tweets
    df = load_data_wrapper(parquet_files)
    # print(df.info(), '\n')
    all_texts = gather_all_texts(df)
    # print(all_texts.shape, '\n')
    train, test = split_data_train_test(df)
    # print(len(train), len(test))
    dls = data_prep_for_lm_model(train, all_texts, tokenizer)
    # create learner for training
    learn = create_learner(dls, model)
    # find learning rate
    lr_min, lr_steep = find_learning_rate(learn)
    print(f'Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}')


if __name__ == "__main__":
    main()
