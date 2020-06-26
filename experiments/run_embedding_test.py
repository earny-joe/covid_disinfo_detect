import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re
import numpy as np
import scipy
import pytest

tokenizer = TweetTokenizer()


def normalizeToken(token):
    lwrcase_tok = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lwrcase_tok.startswith("http") or lwrcase_tok.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    normTweet = re.sub(
        r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet
    )
    normTweet = re.sub(
        r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet
    )
    normTweet = re.sub(
        r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet
    )

    return " ".join(normTweet.split())


def load_narrative_data(filename):
    '''
    Given a file name, loads in narrative data, applies text
    pre-processing, and asserts that there are 20 unique narratives
    in narrative column; returns pandas DataFrame
    '''
    df = pd.read_json(
        filename,
        orient='columns',
        dtype={'id': str}
    )
    # apply text normalization
    df['bert_tweet'] = df['tweet'].apply(
        lambda tweet: normalizeTweet(tweet)
    ).str.lower()
    # make sure we have data for all 20 narratives
    assert len(df['narrative'].unique()) == 20
    return df


def create_embedding_model(model_name):
    '''
    Given string of pretrain embedding available in sentence-transformers
    library, create a SentenceTransformer object to encode embeddings with
    '''
    model = SentenceTransformer(model_name)
    return model


def gen_embeds_narrative_df(tweet_embeddings, df):
    tweet_embeddings_df = pd.DataFrame(tweet_embeddings)
    # change the column names to make them easier to slice
    tweet_embeddings_df.columns = [
        str(col) + '_embed' for col in tweet_embeddings_df.columns
    ]
    # add narrative for each embedding
    tweet_embed_with_narrative = pd.concat(
        [df['narrative'], pd.DataFrame(tweet_embeddings_df)],
        axis=1
    )
    return tweet_embed_with_narrative


def generate_embeddings(model, df, column_name='bert_tweet'):
    '''
    Given a SentenceTransformer model, a pandas DataFrame, and a column name
    (whose default value will take 'bert_tweet'), we'll encode a set of
    (unnormalized) embeddings on tweet text within the dataframe and
    return this dataframe
    '''
    tweets = df[column_name]
    # generate embeddings with model
    tweet_embeddings = model.encode(tweets)
    # create dataframe of tweet embeddings
    tweet_embeddings_df = gen_embeds_narrative_df(tweet_embeddings, df)
    return tweet_embeddings, tweet_embeddings_df


def generate_mean_embeddings(model, df, column_name='bert_tweet'):
    '''
    Given a SentenceTransformer model, a pandas DataFrame, and a column name
    (whose default value will take 'bert_tweet'), we'll encode a set of
    (unnormalized) embeddings on tweet text within the dataframe and return
    a dataframe with the mean embeddings for each narrative
    '''
    tweets = df[column_name]
    # generate embeddings with model
    tweet_embeddings = model.encode(tweets)
    # create dataframe of tweet embeddings
    tweet_embeddings_df = gen_embeds_narrative_df(tweet_embeddings, df)
    # group by narrative and then take mean embedding
    mean_narratives = tweet_embeddings_df.groupby('narrative').mean()
    # group by narrative and then return the standard deviation
    std_narratives = tweet_embeddings_df.groupby('narrative').std()
    assert len(mean_narratives.index) == 20
    assert len(std_narratives.index) == 20
    return mean_narratives, std_narratives


def extract_mean_embedding_values(mean_narratives):
    '''
    Function that takes in mean_narratives pandas DataFrame, drops the
    narratives, which are in the index, and returns only the mean values of
    the emebddings for each narrative
    '''
    mean_narrative_embeddings = (
        mean_narratives.reset_index(drop=True).iloc[:, :].to_numpy()
        )
    return mean_narrative_embeddings


@pytest.fixture
def test_narr_embeds(mean_narr, mean_narr_embed, test_df, test_embed):
    closest_n = int(
        input(
            'What is the number of closest neighbors you would like to test?\n'
        )
    )
    averages = {}
    mean_combo = zip(
        list(mean_narr.index),
        list(mean_narr_embed)
    )
    for narrative, embedding in mean_combo:
        distances = scipy.spatial.distance.cdist(
            [embedding], test_embed, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Narrative:", narrative)
        print(f"\nTop {closest_n} most similar Tweets in teset set:\n")
        cnt = 0
        for idx, distance in results[0:closest_n]:
            obser_narrative = str(test_df['narrative'][idx])
            obser_tweet = test_df['bert_tweet'][idx].strip()
            if str(narrative) == obser_narrative:
                cnt += 1
            averages[narrative] = cnt
            print(
                f'Narratives: {narrative} & {obser_narrative}',
                f'\n{obser_tweet}\n', '(Score: %.4f)' % (1-distance),
                '\n'
                )

    return results, averages


def embedding_average(averages, closest_n):
    narrative_averages = []
    for key in averages.keys():
        percentage = averages[key] / closest_n
        narrative_averages.append(percentage)
    return narrative_averages


def main_test():
    train_filename = 'train_narrative_data.json'
    df_train = load_narrative_data(train_filename)
    model_name = str(
        input('What model for the embeddings do you want to use?\n')
    )
    model = create_embedding_model(model_name)
    # generate dataframes with mean & std, respectively, of the 20 narratives
    mean_narratives, std_narratives = generate_mean_embeddings(model, df_train)
    # generate numpy array containing only the value of the mean embedding
    # without the narrative labels
    mean_narrative_embeddings = extract_mean_embedding_values(mean_narratives)
    # load in test data
    test_filename = 'test_narrative_data_v2.json'
    df_test = load_narrative_data(test_filename)
    # generate embeddings of test data
    tweet_embeddings, tweet_embeddings_df = generate_embeddings(model, df_test)
    # test mean embeddings against test embeddings
    results, averages = test_narr_embeds(
        mean_narratives,
        mean_narrative_embeddings,
        df_test,
        tweet_embeddings
    )
    return results, averages, model_name


def main():
    results, averages, model_name = main_test()
    print(f'{model_name}')
    print(results, '\n')
    print(averages, '\n')
    print(np.mean(embedding_average(averages, 25)))


if __name__ == '__main__':
    main()
