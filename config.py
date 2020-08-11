"""
Config file for generating embeddings component of project pipeline.
"""
from pathlib import Path


NARRATIVES_STOR_PATH = Path.cwd() / 'data/current_centroid_data'
NARRATIVES_FILENAME = 'train_narrative_data_v3.json'

MEAN_NARR_STORAGE_PATH = Path.cwd() / 'data/narrative_data'
MEAN_NARR_FILENAME = 'mean_narrative_data_v3.csv'
MEAN_NARR_DR_FILENAME = 'mean_narrative_data_v3_svddr.csv'
STD_NARR_FILENAME = 'std_narrative_data_v3.csv'

BUCKET_NAME = 'thepanacealab_covid19twitter'
ALL_DATA_FILEPATH = 'gs://thepanacealab_covid19twitter/dailies/*/*_embeddings.csv'
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
