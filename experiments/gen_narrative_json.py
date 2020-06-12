'''
This script takes data gathered for generating cluster centroids for
tweet embeddings from sub-directory of repo's data directory, and
produces a JSON file with all observations and their respectie
narratives.
'''
from pathlib import Path
import pandas as pd
import glob
from tweet_clean import clean_tweet_wrapper
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def path_and_files():
    folder_name = input(
        'Enter name of data sub-folder where JSONs are stored:\n'
    )
    datapath = Path.cwd().parent / 'data' / folder_name
    narrative_files = glob.glob(f'{datapath}/*/*.json')
    return datapath, narrative_files, folder_name


def glob_load(filename):
    df = pd.read_json(
        filename,
        lines=True
    )
    df['narrative'] = str(filename.split('/')[8])
    return df


def sentimentscore(df):
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['tweet'].apply(
        lambda tweet: analyzer.polarity_scores(tweet)['compound']
    )
    return df


def cleandata(df):
    return df[['created_at', 'id', 'tweet', 'sentiment', 'narrative']]


def load_all_narratives(narrative_files):
    for file in narrative_files:
        df = pd.concat([glob_load(file) for file in narrative_files])
        df = sentimentscore(df)
        df = cleandata(df).reset_index(drop=True)
        df['processed_tweet'] = df['tweet'].apply(clean_tweet_wrapper)
        return df


def main():
    datapath, narrative_files, folder_name = path_and_files()
    df = load_all_narratives(narrative_files)
    df.to_json(f'playground_data/{folder_name}.json', orient='columns')
    print(f'Stored {folder_name}.json in playground_data sub-folder')


if __name__ == '__main__':
    main()
