from __future__ import print_function
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from textGenUtils import *
import argparse
import sys
import codecs

# Parsing arguments for RNN definition
argumentParser = argparse.ArgumentParser()
argumentParser.add_argument('-api_key')
argumentParser.add_argument('-api_secret')
argumentParser.add_argument('-access_token')
argumentParser.add_argument('-access_token_secret')
argumentParser.add_argument('-twitter_user', default='GoGoManTweet')
argumentParser.add_argument('-file', default='tweets.npy')
argumentParser.add_argument('-mode', default='train')
argumentParser.add_argument('-weights', default='')
args = vars(argumentParser.parse_args())

API_KEY = args['api_key']
API_SECRET = args['api_secret']
ACCESS_TOKEN = args['access_token']
ACCESS_TOKEN_SECRET = args['access_token_secret']
TWITTER_USER = args['twitter_user']
TWEETS_FILE = args['file']
WEIGHTS = args['weights']

batchSize = 128
layers = 2
maxTwitterLength = 120
layerDimension = 256
epochsToTrain = 20
sequenceLength = 40
sequenceStep = 3
dropout = 0.2
learningRate = 0.01
numOfTweets = 3

# Name of the file that will contain the generated tweets.
generated_tweets_file = 'generated_tweets_dim-{}_layers-{}_epochs-{}.txt'.format(layerDimension, layers, epochsToTrain)
log = open(generated_tweets_file, 'w', 1, "utf-8")

# Setting an encoding for stdout for cross-platform compatibility
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

if TWEETS_FILE == 'tweets.npy':
    # Getting the tweets for specified user
    print('\n\nGetting the tweets:\n')
    save_tweets(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, user_name=TWITTER_USER, file=TWEETS_FILE)

# Creating training data
print('\n\nPreparing the training data:\n')
X, y, vocab_size, index_to_char, sequences = prepare_data(TWEETS_FILE, sequenceLength, log)

# Creating the Network
print('\n\nBuilding the learning model:\n')
model = Sequential()
model.add(LSTM(layerDimension, input_shape=(None, vocab_size), return_sequences=True, dropout=0.2))
for i in range(layers - 1):
    model.add(LSTM(layerDimension, return_sequences=True, dropout=0.2))
model.add(TimeDistributed(Dense(vocab_size)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.01))
print('Model Summary:')
model.summary()

# Training if there is no trained weights specified
if WEIGHTS == '':
    saved_weights = 'checkpoint_layers{}_dim{}'.format(layers, layerDimension)
    checkpoint = ModelCheckpoint(filepath=saved_weights + '_epoch{epoch:02d}.hdf5',
                                 monitor='loss', verbose=1, save_best_only=True, mode='min', period=5)
    model.fit(X, y, batch_size=batchSize, verbose=1, epochs=epochsToTrain, callbacks=[checkpoint])

# Else, loading the trained weights
else:
    model.load_weights(WEIGHTS)

# Generating the tweets
tweets = produce_tweets(model, maxTwitterLength, vocab_size, index_to_char, log, numOfTweets)
print('\n\n')
log.write('\n\n')

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(sequences)
x_val = vectorizer.transform(tweets)
print(pairwise_distances(x_val, Y=tfidf, metric='cosine').min(axis=1).mean())
log.write('{}'.format(pairwise_distances(x_val, Y=tfidf, metric='cosine').min(axis=1).mean()))
