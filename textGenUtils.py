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

    return initialize_arrays(all_chars=all_chars, unique_chars=unique_chars, vocab_size=vocab_size, seq_length=seq_length, step_size=step_size)


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


def initialize_arrays(all_chars, unique_chars, vocab_size, seq_length, step_size):
    index_to_char = {index: char for index, char in enumerate(unique_chars)}
    char_to_index = {char: index for index, char in enumerate(unique_chars)}

    num_of_sequences = (len(all_chars) - seq_length - 1) // step_size + 1
    remaining_chars = (len(all_chars) - seq_length - 1) % step_size

    x = np.zeros((num_of_sequences + (remaining_chars != 0), seq_length, vocab_size))
    y = np.zeros((num_of_sequences + (remaining_chars != 0), seq_length, vocab_size))
    sequences = []
    next_chars = []

    for i in range(0, num_of_sequences + (remaining_chars != 0)):
        if i == num_of_sequences:
            x_sequence = all_chars[(i - 1) * step_size + remaining_chars:((i - 1) * step_size) + remaining_chars + seq_length]
        else:
            x_sequence = all_chars[i * step_size:(i * step_size) + seq_length]
        sequences.append(x_sequence)
        x_sequence_index = [char_to_index[value] for value in x_sequence]
        input_sequence = np.zeros((seq_length, vocab_size))
        for j in range(seq_length):
            input_sequence[j][x_sequence_index[j]] = 1.
            x[i] = input_sequence

        if i == num_of_sequences:
            y_sequence = all_chars[((i - 1) * step_size) + remaining_chars + 1:((i - 1) * step_size) + remaining_chars + seq_length + 1]
        else:
            y_sequence = all_chars[(i * step_size) + 1:(i * step_size) + seq_length + 1]
        next_chars.append(y_sequence)
        y_sequence_index = [char_to_index[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, vocab_size))
        for j in range(seq_length):
            target_sequence[j][y_sequence_index[j]] = 1.
            y[i] = target_sequence

    sequences = np.array(sequences)
    print('Number of sequences: ', len(sequences))
    print('Sequence length: ', seq_length)
    return x, y, vocab_size, index_to_char, sequences


def generate_text(model, length, vocab_size, index_to_char, log, num_of_tweets=3):
    tweets = []
    ending_pattern = re.compile(r"[.!?]")
    for tweet_no in range(num_of_tweets):
        print('\n\nTweet no. {}\n'.format(tweet_no))
        log.write('\n\nTweet no. {}\n'.format(tweet_no))
        # starting with random character
        index = [np.random.randint(vocab_size)]
        print(index_to_char[index[-1]], end="")
        log.write(index_to_char[index[-1]])
        y_char = u''
        y_char += index_to_char[index[-1]]
        x = np.zeros((1, length, vocab_size))
        for i in range(length):
            # appending the last predicted character to sequence
            x[0, i, :][index[-1]] = 1
            index = np.argmax(model.predict(x[:, :i + 1, :])[0], 1)
            predicted_char = index_to_char[index[-1]]
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
