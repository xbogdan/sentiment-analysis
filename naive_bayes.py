from collections import Counter
from io import BytesIO
from tokenize import tokenize


class NaiveBayes:

    def __init__(self):
        self.class_priors = None
        self.classes = None

        self.nb_dict = {}

    @staticmethod
    def calculate_occurences(items):
        nr_items = len(items)
        occurences = dict(Counter(items))
        for key in occurences.keys():
            occurences[key] = occurences[key] / float(nr_items)

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
        return self.calculate_occurences(classes)

    @staticmethod
    def calculate_likelyhood(items, laplace=1):
        """
        Calculates the likelyhood of earch word in a vocabulary
        :param items: list of words
        :param laplace: Laplace smoothing
        :return: dict with the likelyhood value for each word
        """

        nr_items = len(items)
        occurences = dict(Counter(items))
        sum_of_counts = sum(list(occurences.values()))

        for key in occurences.keys():
            occurences[key] = (occurences[key] + laplace) / float(sum_of_counts + nr_items)

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

        g = tokenize(BytesIO(text.encode('utf-8')).readline)
        if to_ignore:
            tokenized_text = [tokval for toknum, tokval, _, _, _ in g if tokval not in to_ignore]
        else:
            tokenized_text = [tokval for toknum, tokval, _, _, _ in g if tokval]

        return tokenized_text

    def identify(self, text):
        """
        Identify the class of which a text belongs to.
        :param text: input text
        :return: class name
        """

        tokenized_text = self.tokenize(text)
        prior = {}
        for label in self.classes:
            prior[label] = self.class_priors[label]
            for word in tokenized_text:
                prior[label] *= self.nb_dict[label][word]

        # TODO handle 50-50 cases
        max_value = 0
        max_class = None
        for label, value in prior.items():
            if value > max_value:
                max_value = value
                max_class = label

        return max_class

    def train(self, vocabularies, classes):
        """"
        When the goal is classifying text, it is better to give the input X in the form of a list of lists containing words.
        X = [
            ['this', 'is', 'a', ...],
            ['i', 'love', 'food', ...],
            ...
        ]

        Y = ['positive', 'positive', 'negative', ...]
        so Y[0] is the class for X[0]
        """

        self.class_priors = self.calculate_priors(classes)

        self.classes = set(classes)
        for label in self.classes:
            self.nb_dict[label] = []

        # combine all examples into one for each class
        # creating a mega document for each class
        for i in range(0, len(classes)):
            self.nb_dict[classes[i]].extend(vocabularies[i])

        # transform the list with all occurences to a dict with relative occurences
        for label in self.classes:
            self.nb_dict[label] = self.calculate_likelyhood(self.nb_dict[label])
