#!/usr/bin/env python3
from naive_bayes import NaiveBayes


def import_lexicon(path):
    keys = set()
    with open(path, encoding='utf-8') as file:
        for line in file:
            if line.startswith(';'):
                continue

            keys.add(line.strip())

    return keys


def main():
    pos_keys = import_lexicon('lexicons/positive-words.txt')
    # neg_keys = import_lexicon('lexicons/negative-words.txt')

    NaiveBayes().train([], ['positive', 'positive', 'negative', 'negative', 'negative'])


if __name__ == '__main__':
    main()
