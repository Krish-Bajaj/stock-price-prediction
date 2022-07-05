import tweepy
import pandas as pd
import config

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def gettweetdata(company):
    # My Twitter API Authentication Variables
    consumer_key = config.consumer_key
    consumer_secret = config.consumer_secret
    access_token = config.access_token
    access_token_secret = config.access_token_secret

    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret)
 
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    tweets = api.search_tweets(company, count=1000)

    data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

    # print(data.head(50))

    import nltk
    nltk.download('vader_lexicon')
    # print(tweets[0].created_at)

    sid = SentimentIntensityAnalyzer()

    listy = []

    for index, row in data.iterrows():
        ss = sid.polarity_scores(row["Tweets"])
        listy.append(ss)

    se = pd.Series(listy)
    data['polarity'] = se.values

    data.to_csv(r'./polarity.csv')
    # print(listy)
    sums = 0.0
    for i in range(1, len(listy)):
        sums = sums + float(listy[i].get('compound'))
    return sums / len(listy)


