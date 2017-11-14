from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from helper_methods import *
import argparse
import sys
import codecs

# Parsing arguments for RNN definition
argumentParser = argparse.ArgumentParser()
argumentParser.add_argument('-file', default='./tweets.txt')
argumentParser.add_argument('-batch_size', type=int, default=50)
argumentParser.add_argument('-layer_num', type=int, default=3)
argumentParser.add_argument('-seq_length', type=int, default=50)
argumentParser.add_argument('-hidden_dim', type=int, default=1024)
argumentParser.add_argument('-generate_length', type=int, default=250)
argumentParser.add_argument('-nb_epoch', type=int, default=20)  # TODO rewrite this shit
argumentParser.add_argument('-num_epoch', type=int, default=100)
argumentParser.add_argument('-mode', default='train')
argumentParser.add_argument('-weights', default='')
args = vars(argumentParser.parse_args())

FILE = args['file']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
NUM_EPOCHS = args['num_epoch']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

fileName = 'generated_text_withHiddenDim{}_and_{}layers.txt'.format(HIDDEN_DIM, LAYER_NUM)
log = open(fileName, 'w', 1, "utf-8")

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Getting the Tweets from a user
print('\n\nGetting the tweets:\n')
get_tweets(file=FILE)

# Creating training data
X, y, VOCAB_SIZE, index_to_char = prepare_data(FILE, SEQ_LENGTH, log)

# Creating the Network
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# Generate some sample before training to know how bad it is!
generate_text(model, args['generate_length'], VOCAB_SIZE, index_to_char, log)

if not WEIGHTS == '':
    model.load_weights(WEIGHTS)
    nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
else:
    nb_epoch = 0

# Training if there is no trained weights specified
if args['mode'] == 'train' or WEIGHTS == '':
    i = 0
    while i < NUM_EPOCHS:
        print('\n\nEpoch: {}\n'.format(nb_epoch))
        log.write('\n\nEpoch: {}\n'.format(nb_epoch))
        model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
        nb_epoch += 1
        i += 1
        generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, index_to_char, log)
        if nb_epoch % 10 == 0:
            model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# Else, loading the trained weights and performing generation only
elif WEIGHTS != '':
    # Loading the trained weights
    model.load_weights(WEIGHTS)
    generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, index_to_char, log)
    print('\n\n')
else:
    print('\n\nNothing to do!')
