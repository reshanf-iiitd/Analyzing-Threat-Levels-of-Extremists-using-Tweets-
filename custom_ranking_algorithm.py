# IMPORTS
import pickle
import pandas as pd
import joblib
import tweepy
import statistics
from textblob import TextBlob
from sklearn import preprocessing
import re
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# To make the code OS independent
fileFolder = os.path.dirname(os.path.realpath('__file__'))


# Returns content of the file
def readFile(fileName):
    file = open(fileName)
    data = file.read()
    file.close()
    return data


# LOAD DATA
def load_data(query):
    tweets = joblib.load(os.path.join(fileFolder, "static/files/the_tweet_dataframe"))
    print("==============================================================================================")
    print(tweets)

    # Load Annotated tweets
    file_name = os.path.join(fileFolder, "static/files/fullyannotated.pkl")
    data_file = open(file_name, 'rb')
    annotated_tweets = pickle.load(data_file)
    tweets['processed_tweet'] = list(annotated_tweets['Tweets'])
    tweets['extremism_cat'] = list(annotated_tweets['Label'])

    # Add userid column to tweets
    user_id = []
    user_name = []
    user_follower = []
    user_tweet_count = []
    tweet_tags = []
    ind = tweets.index
    i = 0
    for tweet in tweets['user']:
        try:
            user_id.append(tweet['id'])
            user_name.append(tweet['name'])
            user_follower.append(tweet['followers_count'])
            user_tweet_count.append(tweet['statuses_count'])
        except:
            tweets = tweets.drop(ind[i])
            i += 1
            continue
        i += 1
    for tweet in tweets['entities']:
        tags = []
        try:
            length = len(tweet['hashtags'])
            for i in range(length):
                tags.append(tweet['hashtags'][i]['text'].lower())
            tweet_tags.append(tags)
        except:
            continue

    tweets['user_id'] = user_id
    tweets['user_name'] = user_name
    tweets['user_follower'] = user_follower
    tweets['user_tweet_count'] = user_tweet_count
    tweets['tweet_tags'] = tweet_tags
    print("==============================================================================================")
    print(tweets)

    tweets = tweets.drop(['source', 'withheld_in_countries',	'limit', 'truncated', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'geo',	'coordinates', 'place',	'contributors',	'retweeted_status'	, 'quoted_status_id'	,
                         'quoted_status_id_str'	, 'quoted_status'	, 'quoted_status_permalink', 'is_quote_status'	, 'quote_count'	, 'reply_count'	, 'retweet_count',	'favorite_count', 'favorited'	, 'retweeted',	'filter_level'	, 'timestamp_ms'	, 'extended_entities'	, 'possibly_sensitive'	, 'display_text_range'	, 'extended_tweet'], axis=1)
    tweets = tweets.dropna()
    print("==============================================================================================")
    print(tweets)

    original_tweets = tweets.copy()

    # Select rows satisfying hashtag
    original_tweets = tweets.copy()
    filtered = []
    for i in original_tweets.index:
        if len(original_tweets.loc[i, 'tweet_tags']) > 0:
            for tag in original_tweets.loc[i, 'tweet_tags']:
                if tag in query:
                    filtered.append(original_tweets.loc[i])
        else:
            continue
    filtered_df = pd.DataFrame(filtered, columns=original_tweets.columns)
    tweets = filtered_df.copy()
    print("==============================================================================================")
    print(tweets)

    user_df = tweets.copy()
    user_df = user_df.drop(['id', 'created_at', 'id_str', 'text', 'user', 'entities'	,
                           'lang', 'extremism_cat', 'processed_tweet', 'tweet_tags'], axis=1)
    user_df = user_df.drop_duplicates()
    the_df = user_df.copy()
    print("==============================================================================================")
    print(user_df)

    # Finding polarity of each tweet by each user, average for each user and append to x_df
    unique_user_polarity = []
    unique_user_subjectivity = []
    for user in user_df['user_id']:
        try:
            polarities = []
            subjectivities = []
            the_user_df = tweets[tweets['user_id'] == user]
            for index, rows in the_user_df.iterrows():
                try:
                    content = rows[0]
                    polarity = TextBlob(content).polarity
                    subjectivity = TextBlob(content).subjectivity
                    polarities.append(polarity)
                    subjectivities.append(subjectivity)
                except:
                    continue
            try:
                avg_polarity = statistics.mean(polarities)
            except:
                avg_polarity = 0
            try:
                avg_subjectivity = statistics.mean(subjectivities)
            except:
                avg_subjectivity = 0
            unique_user_polarity.append(avg_polarity)
            unique_user_subjectivity.append(avg_subjectivity)
        except:
            continue
    the_df['avg polarity'] = unique_user_polarity
    the_df['avg subjectivity'] = unique_user_subjectivity
    print("==============================================================================================")
    print(the_df)

    # LOAD DATA
    tweets2 = tweets.copy()
    print("==============================================================================================")
    print(tweets2)

    list_names_source = []
    for i in range(len(tweets2)):
        try:
            list_names_source.append(tweets2['user'][i]['name'])
        except:
            continue

    dict_mentions = tweets2['entities'].values

    # dict_mentions.shape

    list_mentioneduser = []
    for i in range(len(dict_mentions)):
        try:
            list_mentioneduser.append(dict_mentions[i]['user_mentions'])
        except:
            continue

    mentioned_user = []
    for i in list_mentioneduser:
        temp = []
        if(len(i) > 0):
            for j in range(len(i)):
                temp.append(i[j]['name'])
        mentioned_user.append(temp)
    print("==============================================================================================")
    print(mentioned_user)

    KG_dict = {}
    for i in range(len(list_names_source)):
        KG_dict[list_names_source[i]] = mentioned_user[i]
    print("==============================================================================================")
    print(KG_dict)

    # Create a directed-graph from a dataframe
    G = nx.from_dict_of_lists(KG_dict, create_using=nx.DiGraph())
    plt.figure(figsize=(12, 12))

    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue',
            edge_cmap=plt.cm.Blues, pos=pos, node_size=200, font_size=6)
    plt.show()

    list_of_DC = nx.degree_centrality(G)

    sorted_dict = {}
    sorted_keys = sorted(list_of_DC, key=list_of_DC.get, reverse=True)

    for w in sorted_keys:
        sorted_dict[w] = list_of_DC[w]

    pickle.dump(sorted_dict, open(os.path.join(fileFolder, "static/files/degree_centrality"), "wb"))
    print("==============================================================================================")
    print(sorted_dict)

    mentionindegree = G.in_degree()
    print("==============================================================================================")
    print(mentionindegree)

    def Sort_Tuple(tup):
        return(sorted(tup, key=lambda x: x[1], reverse=True))

    mentionindegree1 = G.out_degree()

    def Sort_Tuple(tup):
        return(sorted(tup, key=lambda x: x[1], reverse=True))

    G.out_edges()  # Representing the name of who mentioning whom

    betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True)

    # Most important connection
    list_mostimpconnection = nx.eigenvector_centrality(G, max_iter=100000)

    sorted_dict_list_mostimpconnection = {}
    sorted_keys = sorted(list_mostimpconnection,
                         key=list_mostimpconnection.get, reverse=True)
    for w in sorted_keys:
        sorted_dict_list_mostimpconnection[w] = list_mostimpconnection[w]

    list_close_centra = nx.closeness_centrality(G)

    sorted_dict_list_close_centra = {}
    sorted_keys = sorted(
        list_close_centra, key=list_close_centra.get, reverse=True)
    for w in sorted_keys:
        sorted_dict_list_close_centra[w] = list_close_centra[w]

    list_between_centra = nx.betweenness_centrality(G)

    sorted_dict_list_between_centra = {}
    sorted_keys = sorted(list_between_centra,
                         key=list_between_centra.get, reverse=True)
    for w in sorted_keys:
        sorted_dict_list_between_centra[w] = list_between_centra[w]
    pickle.dump(sorted_dict_list_between_centra, open(os.path.join(fileFolder, "static/files/betweeness_centrality"), "wb"))

    # Load graph data
    a = np.zeros(the_df.shape[0])
    b = np.zeros(the_df.shape[0])
    the_df['degree_centrality'] = a
    the_df['betweeness_centrality'] = b
    for i in the_df.index:
        try:
            the_df.loc[i, 'degree_centrality'] = sorted_dict[the_df.loc[i, 'user_name']]
        except:
            continue
    for i in the_df.index:
        try:
            the_df.loc[i, 'betweeness_centrality'] = sorted_dict_list_between_centra[the_df.loc[i, 'user_name']]
        except:
            continue

    df = the_df.filter(['user_follower', 'user_tweet_count', 'avg polarity',
                       'avg subjectivity', 'category', 'degree_centrality', 'betweeness_centrality'])
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    counts = df.sum(axis=1, skipna=True)
    df['score'] = counts
    user_names = list(user_df['user_name'])
    df['username'] = user_names

    result = df.sort_values(
        ['score'], ascending=True).drop_duplicates('username')

    resultant_df = result['username']
    ranking = list(resultant_df)
    print(ranking)

    resultant_df.to_json(os.path.join(fileFolder, "static/result.json"))
