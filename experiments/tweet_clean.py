'''
Script used to clean tweet text.
'''
import re
import emoji
from nltk.tokenize import RegexpTokenizer
from spacy.lang.en.stop_words import STOP_WORDS


def newline_remove(text):
    regex = re.compile(r'\n+', re.I)
    text = regex.sub(' ', text)
    return text


def replace_coronavirus(text):
    regex = re.compile(r'(corona[\s]?virus)', re.I)
    return regex.sub('coronavirus', text)


def coronavirus_hashtags(text):
    regex = re.compile(r'#(coronavirus)\b', re.I)
    return regex.sub('xxhash coronavirus', text)


def replace_covid(text):
    regex = re.compile(r'(covid[-\s_]?19)|covid', re.I)
    return regex.sub('covid19', text)


def covid_hashtags(text):
    regex = re.compile(r'#(covid[_-]?(19))', re.I)
    return regex.sub('xxhash covid19', text)


def sarscov2_replace(text):
    regex = re.compile(r'(sars[-]?cov[-]?2)', re.I)
    return regex.sub(r'sarscov2', text)


def emoji_replace(text):
    # first demojize text
    text = emoji.demojize(text)
    regex = re.compile(r'(?<=:)(\S+)(?=:)', re.I)
    for item in regex.finditer(text):
        emojistr = str(item.group())
        replacestr = str(' xxemoji ' + emojistr.replace(r'_', '') + ' ')
        pattern = r"(?::)" + re.escape(emojistr) + r"(?::)"
        text = re.sub(pattern, replacestr,  text)
    return text


def twitterpic_replace(text):
    regex = re.compile(r"pic.twitter.com/\w+", re.I)
    return regex.sub(" xxpictwit ", text)


def youtube_replace(text):
    regex = re.compile(
        r"(https://youtu.be/(\S+))|(https://www.youtube.(\S+))",
        re.I)
    return regex.sub(" xxyoutubeurl ", text)


def url_replace(text):
    regex1 = re.compile(r'(?:http|ftp|https)://(\S+)|(?:www)\.\S+\b', re.I)
    regex2 = re.compile(r'((\b\S+)\.(?:[a-z+]{2,3}))', re.I)
    text = regex1.sub(' xxurl ', text)
    text = regex2.sub(' xxurl ', text)
    return text


def punctuation_replace(text):
    # put spaces between punctuation
    PUNC = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~…–”“’'
    punct = r"[" + re.escape(PUNC) + r"]"
    text = re.sub(
        "(?<! )(?=" + punct + ")|(?<=" + punct + ")(?! )",
        r" ",
        text)
    text = re.sub(r"[^\w\s]", 'xxpunc', text)   # could replace with xxpunc
    # remove any extra whitespace
    text = re.sub(r'[ ]{2,}', ' ', text)
    return text


def clean_tweet_wrapper(text, nltk_tokenize=False, punc_replace=False):
    PUNC = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~…–”“’'
    # removes newline characters from text
    text = newline_remove(text)
    # standardizes all instances of coronavirus in text
    text = replace_coronavirus(text)
    # replaces instances of #coronavirus with special token, xxhashcoronavirus
    text = coronavirus_hashtags(text)
    # standardizes all instances of covid19
    text = replace_covid(text)
    # replaces instances of #covid19 with special token, xxhashcovid19
    text = covid_hashtags(text)
    # standardizes SARS-Cov-2 to sarscov2
    text = sarscov2_replace(text)
    # removes hashtag characters
    text = text.replace(r'#', 'xxhash ')
    # removes @ character
    text = text.replace(r'@', 'xxmention ')
    # replace emojies with special token xxemoji
    text = emoji_replace(text)
    # replace pic.twitter.com links with special token, xxpictwit
    text = twitterpic_replace(text)
    # replace YouTube links with special token, xxyoutubeurl
    text = youtube_replace(text)
    # replace other URLs with special token, xxurl
    text = url_replace(text)
    # if nltk_tokenize True, then use regexp_tokenize from nltk library
    if nltk_tokenize is True:
        tokens = RegexpTokenizer('\\s+', gaps=True).tokenize(text)
        text = ' '.join(
            [''.join(
                [char for char in word if char not in PUNC])
                for word in tokens if word not in STOP_WORDS]
        )
    # if punc_replace set to True, replace all punctuations
    if punc_replace is True:
        text = punctuation_replace(text)
    # remove any unnecessary whitespace
    text = re.sub(r'[ ]{2,}', ' ', text)
    return text.strip()
