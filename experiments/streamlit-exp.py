import streamlit as st
import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import umap
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import viridis


def path_and_files():
    dirpath = Path.cwd()
    datapath = dirpath/'data'/'narrative_tweets'
    narrative_files = glob.glob(
        f'{datapath}/*/*.json'
    )
    return datapath, narrative_files


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


@st.cache
def load_all_narratives(narrative_files):
    for file in narrative_files:
        df = pd.concat([glob_load(file) for file in narrative_files])
        df = sentimentscore(df)
        df = cleandata(df)
        return df


def load_embeddings(filename):
    path = Path().cwd() / 'experiments/playground_data'
    pkl_file = open(
        path / filename, 'rb'
    )
    X = pickle.load(pkl_file)
    return X


def apply_umap(X):
    dr = umap.UMAP(n_components=2, metric='cosine', spread=5)
    dr.fit(X)
    X_dr = dr.transform(X)
    return X_dr


def plot_reduced_2D(X_dr, dr_type):
    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X_dr[:, 0], X_dr[:, 1], alpha=0.8)
    ax.set_title(dr_type+' with 2 components')
    st.pyplot()


@st.cache
def bokeh_df(X_dr, df):
    tweet_embed_df = pd.DataFrame(X_dr, columns=('x', 'y'))
    tweet_embed_df['tweet_id'] = [str(x) for x in df['id']]
    tweet_embed_df['tweet'] = [str(x) for x in df['tweet']]
    tweet_embed_df['narrative'] = [str(x) for x in df['narrative']]
    return tweet_embed_df


def bokeh_plot(df):
    num_colors = len(df.narrative.unique())
    datasource = ColumnDataSource(df)
    color_mapping = CategoricalColorMapper(
        factors=[
            str(x) for x in df.narrative.unique()
        ],
        palette=viridis(num_colors)
    )
    TOOLTIPS = [
        ('Tweet ID', '@tweet_id'),
        ('Tweet', '@tweet'),
        ('Narrative', '@narrative')
    ]
    p = figure(
        plot_width=600,
        plot_height=600,
        tools="pan,wheel_zoom,reset,save",
        tooltips=TOOLTIPS
    )
    p.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='narrative', transform=color_mapping)
    )
    st.bokeh_chart(p, use_container_width=True)


def main():
    datapath, narrative_files = path_and_files()
    df = load_all_narratives(narrative_files)
    st.title('__COVID-19 Misinformation Narratives__')
    filename = st.sidebar.text_input('Enter the file name for embeddings', '')
    if filename:
        X = load_embeddings(filename)
        X_dr = apply_umap(X)
        st.subheader('2D UMAP Projection of Tweet Embeddings')
        tweet_embed_df = bokeh_df(X_dr, df)
        bokeh_plot(tweet_embed_df)


if __name__ == '__main__':
    main()
