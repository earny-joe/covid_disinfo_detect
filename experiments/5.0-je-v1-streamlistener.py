from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream
from twitter_keys_tokens import keys_tokens
import time
import csv
import sys


class TweetListener(StreamListener):
    def __init__(self, api=None):
        self.api = api
        self.filename = "data_" + time.strftime("%Y%m%d") + ".csv"
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
        if "RT @" not in status.text:
            try:
                csvwriter.writerow(
                    [
                        status.created_at, status.id_str, status.user.id_str,
                        status.user.screen_name, status.user.name, status.text,
                        status.reply_count, status.retweet_count,
                        status.favorite_count
                    ]
                )
            except Exception as e:
                print(e)
                pass
        csvfile.close()
        return

    def on_error(self, status_code):
        print("Encountered error with status code:", status_code)
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
    consumer_key = keys_tokens["API_KEY"]
    consumer_secret = keys_tokens["API_SECRET"]
    access_token = keys_tokens["ACCESS_TOKEN"]
    access_token_secret = keys_tokens["ACCESS_SECRET"]
    listen = TweetListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, listen)
    stream.filter(track=queries, languages=["en"])


if __name__ == "__main__":
    start_mining(["covid19, covid-19"])
