'''
streamlit-exp-v2.py
'''
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import seaborn as sns
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CategoricalColorMapper


def gen_random_data():
    '''
    Generates random data to represent a high-dimensional data set.
    dimensions: (1000, 1001)
    '''
    np.random.seed(8)
    # generate column names, data and label names
    column_names = ['col_' + str(x) for x in range(1, 769)]
    data = np.random.uniform(-10, 10, size=(1000, 768))
    label_names = ['label_' + str(x) for x in range(1, 21)]

    # insert numpy array, return pandas dataframe
    df = pd.DataFrame(data, columns=column_names)

    # generate list of labels and randomly insert into pandas dataframe
    labels = np.random.choice(label_names, size=len(df))
    df.insert(loc=0, column='label', value=labels)

    return df


@st.cache
def gather_umap_data():
    '''
    Gather UMAP data from pseudo-embeddings, filename1 and filename2
    will be updated in near future to reflect name in config file
    (which will most likely have to do with some sort of database
    because of the size of the embeddings and their UMAP version)
    '''
    cwd = Path.cwd()
    filename1 = 'all_umap_pseudo_embeddings.json'
    filename2 = 'centers_umap_pseudo_embeddings.json'
    path_data = cwd/'data'/'current_centroid_data'
    df_all = pd.read_json(
        path_data/filename1,
        orient='columns'
    )
    df_centers = pd.read_json(
        path_data/filename2,
        orient='columns'
    )
    return df_all, df_centers


def bokeh_plot(df_all):
    label_count = len(df_all['label'].unique())
    color_scheme = sns.color_palette(
        'Paired',
        label_count
    ).as_hex()
    datasource_all = ColumnDataSource(df_all)
    # datasource_center = ColumnDataSource(df_centers)
    color_mapping = CategoricalColorMapper(
        factors=[
            str(x) for x in df_all['label'].unique()
        ],
        palette=color_scheme
    )
    TOOLTIPS = [
        ('X', '@x'),
        ('Y', '@y'),
        ('Label', '@label')
    ]
    p = figure(
        plot_width=800,
        plot_height=800,
        tools="pan,wheel_zoom,reset,save",
        tooltips=TOOLTIPS
    )
    p.circle(
        'x',
        'y',
        source=datasource_all,
        color=dict(field='label', transform=color_mapping),
        size=8,
        line_color='grey',
        line_alpha=0.5
    )
    st.bokeh_chart(p, use_container_width=True)


def main():
    df_all, df_centers = gather_umap_data()
    st.title('Test Visualization App')
    option_all = st.sidebar.selectbox(
        'See all data?',
        ('True', 'False')
    )
    option_center = st.sidebar.selectbox(
        'See centroid data?',
        ('True', 'False')
    )
    bokeh_plot(df_all)

    if option_all == 'True':
        st.dataframe(df_all)
    if option_center == 'True':
        st.dataframe(df_centers)


if __name__ == '__main__':
    main()
