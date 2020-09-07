"""
Config file for generating embeddings component of project pipeline.
"""
from pathlib import Path


COLS_SELECT = [
    'created_at',
    'id_str',
    'user',
    'source',
    'is_quote_status',
    'retweet_count',
    'favorite_count',
    'lang',
    'full_text'
]
COLS_INTEREST = [
    'created_at',
    'id_str',
    'user_id_str',
    'source',
    'is_quote_status',
    'retweet_count',
    'favorite_count',
    'lang',
    'full_text'
]

VERSION = '20200826'
CENTROID_STOR_PATH = Path.cwd() / 'data/current_centroid_data'
CENTROID_FILENAME = f'centroid_data_{VERSION}.json'

NARR_STORAGE_PATH = Path.cwd() / 'data/narrative_data'
MEAN_NARR_FILENAME = f'mean_narrative_data_{VERSION}.csv'
MEAN_NARR_DR_FILENAME = f'mean_narrative_data_{VERSION}_svddr.csv'
MEDIAN_NARR_FILENAME = f'median_narrative_data_{VERSION}.csv'
MEDIAN_NARR_DR_FILENAME = f'median_narrative_data_{VERSION}_svddr.csv'
STD_NARR_FILENAME = f'std_narrative_data_{VERSION}.csv'

BUCKET_NAME = 'my_sm_project_data'
ALL_DATA_FILEPATH = 'gs://my_sm_project_data/dailies/*/*_embeddings.csv'
EMBED_MODEL_NAME = 'distilbert-base-nli-stsb-mean-tokens'
MODEL_DIRECT_PATH = Path.cwd() / 'data/models'
MODEL_FILE_NAME = 'svd_embedding.pkl'
SEED_VALUE = 8

TOOLTIPS_TWEETS = [
    ('Tweet ID', '@tweet_id'),
    ('Created', '@created_at'),
    ('Tweet', '@tweet'),
    ('Narrative', '@narrative')
]
