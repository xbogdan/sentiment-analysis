#!/usr/bin/env python3
import os
from random import shuffle

from naive_bayes import NaiveBayes


def import_lexicon(path):
    keys = set()
    with open(path, encoding='utf-8', errors='ignore') as file:
        for line in file:
            if line.startswith(';'):
                continue

            keys.add(line.strip())

    return keys


def read_from_dir(directory, file_type='.txt', limit=800):
    context_list = []

    ll = os.listdir(directory)
    shuffle(ll)

    file_list = ll[:limit] if limit else os.listdir(directory)
    for filename in file_list:
        if file_type and filename.endswith(file_type) or not file_type:
            with open(f'{directory}/{filename}', 'r', encoding='utf-8', errors='ignore') as file:
                context_list.append(file.read())

    test_list = []
    test = ll[limit:] if limit else os.listdir(directory)
    for filename in test:
        if file_type and filename.endswith(file_type) or not file_type:
            with open(f'{directory}/{filename}', 'r', encoding='utf-8', errors='ignore') as file:
                test_list.append(file.read())

    return context_list, test_list


def main():
    # pos_keys = import_lexicon('training_sets/lexicons/positive-words.txt')
    # neg_keys = import_lexicon('training_sets/lexicons/negative-words.txt')

    pos_train_set, pos_test_set = read_from_dir('training_sets/reviews/pos1')
    neg_train_set, neg_test_set = read_from_dir('training_sets/reviews/neg1')
    print(len(pos_train_set), len(pos_test_set))
    print(len(neg_train_set), len(neg_test_set))

    nb = NaiveBayes()
    nb.train(vocabularies=pos_train_set+neg_train_set,
             classes=['positive' for _ in range(0, len(pos_train_set))] +
                     ['negative' for _ in range(0, len(neg_train_set))])

    # postive test
    pos_returns = {'positive': 0, 'negative': 0}
    for value in pos_test_set:
        # print(nb.identify(pos_value))
        if nb.identify(value) == 'positive':
            pos_returns['positive'] += 1
        else:
            pos_returns['negative'] += 1
    print('positve', pos_returns['positive'] * 100/(pos_returns['positive']+pos_returns['negative']), '% correct')

    # negative test
    pos_returns = {'positive': 0, 'negative': 0}
    for value in neg_test_set:
        # print(nb.identify(pos_value))
        if nb.identify(value) == 'negative':
            pos_returns['negative'] += 1
        else:
            pos_returns['positive'] += 1
    print('negative', pos_returns['negative'] * 100 / (pos_returns['positive'] + pos_returns['negative']), '% correct')


if __name__ == '__main__':
    main()
