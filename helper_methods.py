from __future__ import print_function
import numpy as np
import tweepy


# prepare data
def prepare_data(file, seq_length, log):
    # all_chars contains all the examples (characters), unique_chars acts as a features holder
    all_chars = open(file, 'r', encoding='UTF-8').read()  # open file and read contents into 'all_chars' array, where each element is 1 character
    unique_chars = list(set(all_chars))  # same as 'all_chars', but contains only unique chars -- those repeating in 'all_chars' are discarded
    vocab_size = len(unique_chars)  # size of our vocabulary - unique chars

    print('Data length: {} characters'.format(len(all_chars)))
    log.write('Data length: {} characters\n'.format(len(all_chars)))
    print('Vocabulary size: {} characters'.format(vocab_size))
    log.write('Vocabulary size: {} characters\n\n'.format(vocab_size))

    index_to_char = {index: char for index, char in enumerate(unique_chars)}
    char_to_index = {char: index for index, char in enumerate(unique_chars)}

    X = np.zeros((len(all_chars) // seq_length, seq_length, vocab_size))
    y = np.zeros((len(all_chars) // seq_length, seq_length, vocab_size))
    for i in range(0, len(all_chars) // seq_length):
        X_sequence = all_chars[i * seq_length:(i + 1) * seq_length]
        X_sequence_index = [char_to_index[value] for value in X_sequence]
        input_sequence = np.zeros((seq_length, vocab_size))
        for j in range(seq_length):
            input_sequence[j][X_sequence_index[j]] = 1.
            X[i] = input_sequence

        y_sequence = all_chars[i * seq_length + 1:(i + 1) * seq_length + 1]
        y_sequence_index = [char_to_index[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, vocab_size))
        for j in range(seq_length):
            target_sequence[j][y_sequence_index[j]] = 1.
            y[i] = target_sequence
    return X, y, vocab_size, index_to_char


def generate_text(model, length, vocab_size, index_to_char, log):
    # starting with random character
    index = [np.random.randint(vocab_size)]
    y_char = [index_to_char[index[-1]]]
    X = np.zeros((1, length, vocab_size))
    for i in range(length):
        # appending the last predicted character to sequence
        X[0, i, :][index[-1]] = 1
        print(index_to_char[index[-1]], end="")
        log.write(index_to_char[index[-1]])
        index = np.argmax(model.predict(X[:, :i + 1, :])[0], 1)
        y_char.append(index_to_char[index[-1]])
    return ''.join(y_char)


def get_tweets(uid=713035525, file='tweets.txt'):
    auth = tweepy.OAuthHandler('Sa9K98ptakr1H4KBsJEVxMrYI', 'Wls5SV7jl49F99fIy3OpgQCucJDAwYrQBZ0r7JRXDQLe2VTzvP')
    auth.set_access_token('899978729396547590-prRio0cVA3dbAEZBZR16yJJAfTnWuMN', 'sSEZyg005EhoHd81OkE61QfwIv1bWi33PFQrCBmDkvU9n')
    api = tweepy.API(auth)
    i = 0

    try:
        tweets_file = open(file, 'x', 1, "utf-8")
        tweets = tweepy.Cursor(api.user_timeline, user_id=uid, tweet_mode='extended').items()
        for status in tweets:
            tweets_file.write(status._json['full_text'] + '\n')
            i += 1
        print('{} tweets have been saved to a file {}\n'.format(i, file))
    except FileExistsError:
        print('File {} already exists, using it as a source of tweets.'.format(file))
        # tweets_file = open('tweets.txt', 'r', 1, "utf-8")
        # print('\n\nTweet with id ' + status._json['id_str'] + ':' + status._json['full_text'] + '\n')
