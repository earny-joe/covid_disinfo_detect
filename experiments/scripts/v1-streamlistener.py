import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream
import time
import csv
import sys
import os


class TweetListener(StreamListener):
    def __init__(self, api=None):
        self.api = api
        self.filename = "data_" + time.strftime("%Y%m%d") + ".csv"
        self.num_tweets = 0
        csvfile = open(self.filename, "w")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                "created_at", "id", "user_id", "username", "name", "tweet",
                "replies_count", "retweets_count", "likes_count"
            ]
        )

    def on_status(self, status):
        csvfile = open(self.filename, "a")
        csvwriter = csv.writer(csvfile)
        if hasattr(status, 'retweeted_status'):
            pass
        elif hasattr(status, 'extended_tweet') and self.num_tweets < 100:
            try:
                csvwriter.writerow(
                    [
                        status.created_at, status.id_str, status.user.id_str,
                        status.user.screen_name, status.user.name,
                        status.extended_tweet.full_text, status.reply_count,
                        status.retweet_count, status.favorite_count
                    ]
                )
                self.num_tweet += 1
            except tweepy.TweepError as e:
                print(e)
                pass
        elif self.num_tweets < 100:
            try:
                csvwriter.writerow(
                    [
                        status.created_at, status.id_str, status.user.id_str,
                        status.user.screen_name, status.user.name,
                        status.text, status.reply_count,
                        status.retweet_count, status.favorite_count
                    ]
                )
                self.num_tweet += 1
            except Exception:
                csvfile.close()
        return

    def on_error(self, status_code):
        print(f"Encountered error with status code: {status_code}")
        if status_code == 401:
            return False

    def on_delete(self, status_id, user_id):
        print("Delete notice")
        return

    def on_limit(self, track):
        print("Rate limited, continuing")
        return True

    def on_timeout(self):
        print(sys.stderr, "Timeout...")
        time.sleep(10)
        return


def start_mining(queries):
    consumer_key = os.environ.get("API_KEY")
    consumer_secret = os.environ.get("API_SECRET")
    access_token = os.environ.get("ACCESS_TOKEN")
    access_token_secret = os.environ.get("ACCESS_SECRET")
    listen = TweetListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, listen)
    stream.filter(track=queries, languages=["en"], tweet_mode='extended')


if __name__ == "__main__":
    lst = [item for item in input("Enter the list items : ").split()]
    start_mining(lst)
