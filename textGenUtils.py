from __future__ import print_function
import numpy as np
import tweepy
import re


# Prepare the data. The characters are read from the *file* by *seq_length* characters and then an information about
# the processed data is written to log file as well as to standard output
def prepare_data(file, seq_length, log, step_size=3):
    # all_chars variable contains all the examples (characters), unique_chars acts as a features holder for our RNN
    tweets = regularize_tweets(file)
    all_chars = u' '.join(tweets)  # join all tweets with a space, into 'all_chars' array
    # only unique chars from 'all_chars' - those repeating in 'all_chars' are discarded. This is our vocabulary.
    unique_chars = sorted(list(set(all_chars)))
    vocab_size = len(unique_chars)  # size of our vocabulary - unique chars

    print('Data length: {} characters'.format(len(all_chars)))
    log.write('Data length: {} characters\n'.format(len(all_chars)))
    print('Vocabulary size: {} characters'.format(vocab_size))
    log.write('Vocabulary size: {} characters\n\n'.format(vocab_size))

    return initialize_arrays(all_chars, unique_chars, vocab_size, seq_length, step_size, log)


# Remove any "non-regular" chars, like emojis, wild spaces, URLs and so on
def regularize_tweets(file):
    tweets = np.load(file)
    remove_urls_regex = re.compile(r"[:\s]*http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    tweets = [remove_urls_regex.sub('', t) for t in tweets]
    remove_weird_chars = re.compile(r"(?:[ ]?-[ ]?)+|(?:&\w+)|(?:[:;][\S])|[\n\t]+")
    tweets = [remove_weird_chars.sub('', t) for t in tweets]

    # https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    remove_emoji = re.compile("["
                              u"\U0001F600-\U0001F64F"  # emoticons
                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                              u"\U0001F680-\U0001F6FF"  # transport & map symbols
                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                              "]+", flags=re.UNICODE)
    tweets = [remove_emoji.sub('', t) for t in tweets]
    remove_redundant_spaces = re.compile(r"[ ]{2,}")
    tweets = [remove_redundant_spaces.sub(' ', t) for t in tweets]
    tweets = [t.lower() for t in tweets]
    return tweets


def initialize_arrays(all_chars, unique_chars, vocab_size, seq_length, step_size, log):
    index_to_char = {index: char for index, char in enumerate(unique_chars)}
    char_to_index = {char: index for index, char in enumerate(unique_chars)}

    # divide all tweets into sequences, each starting from length of step_size from previous one
    # next_chars are characters that follow in each sequence
    sequences = []
    next_chars = []
    for i in range(0, len(all_chars) - seq_length, step_size):
        sequences.append(all_chars[i:i + seq_length])
        next_chars.append(all_chars[i + seq_length])
    sequences = np.array(sequences)
    next_chars = np.array(next_chars)

    # one-hot encode - binary values, true for index where a current char is specified
    x = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
    y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for j, char in enumerate(sequence):
            x[i, j, char_to_index[char]] = 1
        y[i, char_to_index[next_chars[i]]] = 1

    print('Number of sequences: ', len(sequences))
    log.write('Number of sequences: {}\n'.format(len(sequences)))
    print('Sequence length: ', seq_length)
    log.write('Sequence length: {}'.format(seq_length))
    return x, y, vocab_size, index_to_char, sequences


def produce_tweets(model, length, vocab_size, index_to_char, log, num_of_tweets=3, temperature=0.2):
    tweets = []
    ending_pattern = re.compile(r"[.!?]")
    for tweet_no in range(num_of_tweets):
        print('\n\nTweet no. {} and temperature of {}\n'.format(tweet_no, temperature))
        log.write('\n\nTweet no. {} and temperature of {}\n'.format(tweet_no, temperature))
        # starting with random character
        index = [np.random.randint(vocab_size)]
        print(index_to_char[index[-1]], end="")
        log.write(index_to_char[index[-1]])
        y_char = u''
        y_char += index_to_char[index[-1]]
        x = np.zeros((1, length, vocab_size))
        for i in range(length):
            # appending the last predicted character to sequence
            x[0, i, :][index] = 1
            predictions = model.predict(x[:, :i + 1, :])[0]
            index = sample(predictions, temperature)
            predicted_char = index_to_char[index]
            print(predicted_char, end="")
            log.write(predicted_char)
            y_char += predicted_char

            # check if the last 20 chars of a tweet end with one of [.!?]
            # if yes, we end the tweet generation
            if i >= length - 20 and ending_pattern.match(predicted_char) is not None:
                break

        tweets.append(y_char)
    return tweets


def save_tweets(api_key, api_secret, access_token, access_token_secret, user_name, file):
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    i = 0
    tweets_array = []

    tweets = tweepy.Cursor(api.user_timeline, screen_name=user_name, tweet_mode='extended').items()
    try:
        tweets_file = open('received_tweets.txt', 'x', 1, "utf-8")
    except FileExistsError:
        tweets_file = open('received_tweets.txt', 'w', 1, "utf-8")
    for status in tweets:
        tweets_file.write(status.full_text + '\n')
        tweets_array.append(status.full_text)
        i += 1
    np.save(file=file, arr=tweets_array)
    print('{} tweets have been saved to a file {}.'.format(i, file))
    print('You can see received tweets in file received_tweets.txt.\n')


# from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
