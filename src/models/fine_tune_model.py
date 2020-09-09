"""
Beginning work towards script that'll fine-tune model for tweets
"""
from sentence_transformers import SentencesDataset, InputExample
import utils
import pandas as pd
import settings.config as config


def load_daily_data(day):
    """
    TBD
    """
    bucket_path = (
        f'gs://{config.BUCKET_NAME}/dailies/{day}/{day}_twitter_data.parquet'
    )
    # with only first 10k tweets for testing
    df = pd.read_parquet(bucket_path)[:10000]
    # gather lengths of tweets
    df['tweet_length'] = df['full_text'].apply(lambda x: len(x))
    # remove tweets that are abnormally long
    df = df[df['tweet_length'] <= 330].reset_index(drop=True)
    # then drop column (we don't need it any further)
    df = df.drop(columns='tweet_length')
    # prep for embeddings
    df = utils.clean_for_embeddings(df)

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

    day = input('Pick date to test with (YYYY-MM-DD)\n')
    df = load_daily_data(day)

    train_dataset = gather_training_examples(df, model)
    print(train_dataset)


if __name__ == "__main__":
    main()
