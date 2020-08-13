import pandas as pd
import streamlit as st
import seaborn as sns
from google.cloud import storage
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CategoricalColorMapper
from settings.config import BUCKET_NAME, NARR_STORAGE_PATH, MEDIAN_NARR_DR_FILENAME, TOOLTIPS_TWEETS


def available_dates(bucket_name=BUCKET_NAME):
    '''
    Gathers the dates of tweet-related parquet files already stored in GCS
    '''
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix='dailies/')
    files = [
        str(i).split(',')[1].strip() for i in blobs
        if str(i).split(',')[1].endswith('_embeddings_svddr.parquet')
    ]
    available_dates = [
        i.split('/')[1] for i in files
    ]
    return available_dates


@st.cache
def load_daily_data(day):
    """
    Temp function to load test data for Streamlit visual
    """
    filename = (
        f'gs://my_sm_project_data/dailies/{day}/{day}_embeddings_svddr.parquet'
    )

    df_day = pd.read_parquet(
        f'{filename}'
    )

    return df_day


def load_narrative_data():
    """
    Loads in 2D median embeddings for various misinfo narratives
    """
    df_narr = pd.read_csv(
        f'{NARR_STORAGE_PATH}/{MEDIAN_NARR_DR_FILENAME}',
        index_col=False
    )
    return df_narr


def color_scheme_misinfo_narratives(df_narr):
    """
    Generates color schemes for various misinfo narratives
    """
    label_count = len(df_narr['narrative'].unique())

    color_scheme = sns.color_palette(
        'bright',
        label_count
    ).as_hex()

    color_mapping = CategoricalColorMapper(
        factors=[
            str(x) for x in df_narr['narrative'].unique()
        ],
        palette=color_scheme
    )

    return color_mapping


def sidebar_settings():
    """Add selection section for setting setting the max-width and padding
    of the main block container"""
    st.sidebar.header("Visualization Width")
    max_width_100_percent = st.sidebar.checkbox("Max-width?", False)
    if not max_width_100_percent:
        max_width = st.sidebar.slider("Select max-width in px", 100, 2000, 1200, 100)
    else:
        max_width = 2000

    _set_block_container_style(max_width, max_width_100_percent)
    return max_width


def _set_block_container_style(
    max_width: int = 1200, max_width_100_percent: bool = False
):
    if max_width_100_percent:
        max_width_str = "max-width: 95%;"
    else:
        max_width_str = f"max-width: {max_width}px;"
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
</style>
""",
        unsafe_allow_html=True,
    )


def bokeh_plot(df_day, df_narr, max_width):
    """
    Given pandas DataFrame of specific day's data, renders the 2d
    embeddings for that day
    """
    source_day = ColumnDataSource(df_day)
    source_misinfo = ColumnDataSource(df_narr)
    color_mapping = color_scheme_misinfo_narratives(df_narr)

    plot = figure(
        plot_width=max_width,
        plot_height=1000,
        title='COVID-related Tweets & COVID Misinfo Narratives',
        tooltips=TOOLTIPS_TWEETS
    )

    plot.circle(
        'x',
        'y',
        source=source_day,
        fill_alpha=0.7,
        fill_color='steelblue',
        line_alpha=0.3,
        line_color='black',
        line_width=0.3
    )
    # plot.add_tools(HoverTool(tooltips=config.TOOLTIPS_TWEETS))

    plot.diamond(
        'x',
        'y',
        source=source_misinfo,
        size=12,
        color=dict(field='narrative', transform=color_mapping),
        line_color='black'
    )

    plot.title.text_font = 'tahoma'
    plot.title.text_font_style = 'italic'
    plot.toolbar.autohide = True

    st.bokeh_chart(
        plot,
        use_container_width=False
    )


def main():
    """
    Main application
    """
    st.title('2D Embedding Visualization App (v3)')
    max_width = sidebar_settings()

    dates = available_dates()
    st.sidebar.header('Access Tweet Data')
    day = st.sidebar.selectbox(
        'What day would you like to explore?',
        dates
    )
    num_tweets = st.sidebar.slider(
        'Number of Tweets to Sample',
        min_value=1000,
        max_value=100000,
        value=10000,
        step=250
    )

    df_day = load_daily_data(day)
    df_day_sample = df_day.sample(n=num_tweets, random_state=8)
    df_narr = load_narrative_data()

    bokeh_plot(df_day_sample, df_narr, max_width)


if __name__ == "__main__":
    main()
    # load_narrative_data()
