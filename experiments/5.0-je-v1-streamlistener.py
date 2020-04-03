#!/usr/bin/env python
# coding: utf-8

# # _Experiment: Tweepy `StreamListener`_
# 
# **TL;DR** --> Begin developing Tweepy `StreamListener` to be able to stream Tweets.

# In[1]:

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, Stream
import time
import csv
import sys

# In[2]:

# create a streamer object
class TweetListener(StreamListener):
    
    # define function that is initialized when the miner is called
    def __init__(self, api = None):
        
        # sets API
        self.api = api
        # create filename
        self.filename = "data_" + time.strftime("%Y%m%d") + ".csv"
        # create a new file with above filename
        csvfile = open(self.filename, "w")
        
        # create csv writer
        csvwriter = csv.writer(csvfile)
        
        # write single row with the headers of the columns
        csvwriter.writerow([
            "created_at", "id", "user_id", "username",
            "name", "tweet", "replies_count", "retweets_count",
            "likes_count"  
        ])
        
    # when a tweet appears
    def on_status(self, status):
        
        # open csv file created previously
        csvfile = open(self.filename, "a")
        # create csv writer
        csvwriter = csv.writer(csvfile)
        
        # if the tweet is not a retweet
        if not "RT @" in status.text:
            # try to
            try:
                # write the tweet's information to the csv file
                csvwriter.writerow([
                    status.created_at, status.id_str, status.user.id_str, status.user.screen_name,
                    status.user.name, status.text, status.reply_count, status.retweet_count, 
                    status.favorite_count
                ])
            # if some error occurs
            except Exception as e:
                # print the error 
                print(e)
                # and continue
                pass
        
        # close the csv file
        csvfile.close()
        
        # return nothing
        return 
    
    # when an error occurs
    def on_error(self, status_code):
        
        #print the error code
        print("Encountered error with status code:", status_code)
        
        # if the error code is 401, whish is error for bad credentials
        if status_code == 401:
            # end the stream
            return False
        
    # when a deleted tweet appears
    def on_delete(self, status_id, user_id):
        
        # print message
        print("Delete notice")
        # return nothing
        return
    
    # when reach the rate limit
    def on_limit(self, track):
        
        # print rate limiting error
        print("Rate limited, continuing")
        # continue mining tweets
        return True
    
    # when timed out
    def on_timeout(self):
        
        # print timeout message
        print(sys.stderr, "Timeout...")
        
        # wait 10 seconds
        time.sleep(10)
        
        # return nothing
        return

# ## _Create Wrapper for Tweet Miner_

# In[4]:

from twitter_keys_tokens import keys_tokens

# In[5]:

# create mining function
def start_mining(queries):
    """
    Given list of strings, returns tweets containing those strings
    """
    
    # variables that contain user credentials
    consumer_key = keys_tokens["API_KEY"]
    consumer_secret = keys_tokens["API_SECRET"]
    access_token = keys_tokens["ACCESS_TOKEN"]
    access_token_secret = keys_tokens["ACCESS_SECRET"]
    
    # create a listener
    listen = TweetListener()
    
    # create authorization info
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    
    # create a stream object with listener and authorization
    stream = Stream(auth, listen)
    
    # run the stream object using the user defined queries
    stream.filter(track=queries, languages=["en"])

    
if __name__ == "__main__":
    start_mining(["covid19, covid-19"])
