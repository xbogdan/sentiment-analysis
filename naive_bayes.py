from collections import Counter
from io import BytesIO
from tokenize import tokenize

from decimal import Decimal


class NaiveBayes:

    def __init__(self):
        self.vocabulary_dict = {}
        self.class_priors = None
        self.classes = None

        self.nb_dict = {}
        self.mega_doc = {}

    @staticmethod
    def calculate_probabilities(items):
        nr_items = len(items)
        occurences = dict(Counter(items))
        for key in occurences.keys():
            occurences[key] = occurences[key] / Decimal(nr_items)

        return occurences

    @staticmethod
    def mark_as_negative(text):
        """
        Mark words in negative sentences as negative words.
        It appends NOT_ to each word prceded by a negation.
        For example "I didn't like the movie, but I ..." transforms to "I didn't NOT_like NOT_this NOT_movie, but I" 
        :param text: text
        :return: text words marked as negative
        """

        # TODO implement
        return text

    @staticmethod
    def remove_duplicates(text):
        """
        Boolean Naive Bayes.
        Removes duplicate words.
        Should be applied for each document in the mega document
        Should be applied after mark_as_negative
        """

        return text

    def calculate_priors(self, classes):
        return self.calculate_probabilities(classes)

    @staticmethod
    def calculate_likelyhood(items, vocabulary_size, laplace=1):
        """
        Calculates the likelyhood of earch word in a vocabulary
        :param vocabulary_size: 
        :param items: list of words
        :param laplace: Laplace smoothing
        :return: dict with the likelyhood value for each word
        """

        nr_items = len(items)
        occurences = dict(Counter(items))

        for key in occurences.keys():
            # print(key, ' - ', occurences[key]+laplace, nr_items+ vocabulary_size)
            occurences[key] = (occurences[key] + laplace) / Decimal(vocabulary_size + nr_items)
            # print(key, occurences[key])

        return occurences

    @staticmethod
    def tokenize(text, to_ignore=None):
        """
        Tokenize a text based on specific rules
        It depends on the type of text and the source. 
        Example: html, twitter posts, text from a book, etc.
        :param text: text 
        :param to_ignore: a list of tokens to ignore
        :return: a list of words
        """
        # byte_obj = BytesIO(text.strip().encode('utf-8'))
        # g = tokenize(byte_obj.readline)
        #
        # import ipdb; ipdb.set_trace()
        # if to_ignore:
        #     tokenized_text = [tokval for toknum, tokval, _, _, _ in g if tokval not in to_ignore]
        # else:
        #     tokenized_text = [tokval for toknum, tokval, _, _, _ in g if tokval]
        tokenized_text = text.strip()
        tokenized_text = tokenized_text.replace(',', ' ')
        tokenized_text = tokenized_text.replace('.', ' ')
        tokenized_text = tokenized_text.replace(';', ' ')
        tokenized_text = tokenized_text.replace('"', ' ')
        tokenized_text = tokenized_text.replace('\'', ' ')
        tokenized_text = tokenized_text.replace(':', ' ')
        tokenized_text = tokenized_text.replace('{', ' ')
        tokenized_text = tokenized_text.replace('}', ' ')
        tokenized_text = tokenized_text.replace(')', ' ')
        tokenized_text = tokenized_text.replace('(', ' ')
        tokenized_text = tokenized_text.replace('/', ' ')
        tokenized_text = tokenized_text.replace('?', ' ')
        tokenized_text = tokenized_text.split()
        return tokenized_text

    def identify(self, text):
        """
        Identify the class of which a text belongs to.
        :param text: input text
        :return: class name
        """

        tokenized_text = self.tokenize(text.lower())
        prior = {}
        for label in self.classes:
            prior[label] = self.class_priors[label]
            for word in tokenized_text:
                if word in self.vocabulary_dict:
                    prior[label] *= Decimal(self.vocabulary_dict[word][label])

        # TODO handle 50-50 cases
        max_value = 0
        max_class = None
        for label, value in prior.items():
            if value > max_value:
                max_value = value
                max_class = label

        return max_class

    def train(self, vocabularies, classes):
        """
        When the goal is classifying text, it is better to give the input vocabularies 
        in the form of a list of lists containing words.
        X = [
            ['this', 'is', 'a', ...],
            ['i', 'love', 'food', ...],
            ...
        ]

        classes = ['positive', 'positive', 'negative', ...]
        so classes[0] is the class for vocabularies[0]
        """

        # split in words each text
        for i, v in enumerate(vocabularies):
            vocabularies[i] = self.tokenize(v.lower())

        # class probability
        self.class_priors = self.calculate_priors(classes)

        self.classes = set(classes)

        # creates mega documents per class
        for label in self.classes:
            self.mega_doc[label] = []

        # combine all examples into one for each class
        # creating a mega document for each class
        vocabulary_size = 0

        for i in range(0, len(classes)):
            self.mega_doc[classes[i]].extend(vocabularies[i])

        vocabulary = []
        for label in self.classes:
            vocabulary.extend(self.mega_doc[label])

        vocabulary = set(vocabulary)
        vocabulary_size += len(vocabulary)
        # print('Vocabulary size', vocabulary_size)

        # transform the list with all occurences to a dict with relative occurences
        for label in self.classes:
            self.nb_dict[label] = dict(Counter(self.mega_doc[label]))

        for word in vocabulary:
            self.vocabulary_dict[word] = {}
            for label in self.classes:
                self.vocabulary_dict[word][label] = None

        for label in self.classes:
            for word in vocabulary:
                word_occ = self.nb_dict[label][word] if word in self.nb_dict[label] else 0
                # print(f'{word}|{label} = ({word_occ} + 1) / ({len(self.mega_doc[label])} + {vocabulary_size})')
                val = Decimal((word_occ + 1) / Decimal(vocabulary_size + len(self.mega_doc[label])))
                self.vocabulary_dict[word][label] = val

