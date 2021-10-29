import ast
from nltk.stem.snowball import SnowballStemmer
from code.preprocessing.preprocessor import Preprocessor


class Stemmer(Preprocessor):
    def __init__(self, input_columns, output_column):
        super().__init__([input_columns], output_column)

    def _set_variables(self, inputs):
        self._arabic_stemmer = SnowballStemmer(language='arabic')
        self._english_stemmer = SnowballStemmer(language="english")
        self._french_stemmer = SnowballStemmer(language='french')
        self._german_stemmer = SnowballStemmer(language='german')
        self._spanish_stemmer = SnowballStemmer(language="spanish")
        self._danish_stemmer = SnowballStemmer(language='danish')
        self._dutch_stemmer = SnowballStemmer(language='dutch')
        self._hungarian_stemmer = SnowballStemmer(language="hungarian")
        self._italian_stemmer = SnowballStemmer(language='italian')
        self._norwegian_stemmer = SnowballStemmer(language='norwegian')
        self._portuguese_stemmer = SnowballStemmer(language='portuguese')
        self._romanian_stemmer = SnowballStemmer(language="romanian")
        self._russian_stemmer = SnowballStemmer(language='russian')
        self._swedish_stemmer = SnowballStemmer(language='swedish')

    def _get_values(self, inputs):
        inputs = inputs[0]
        tweets_stemmed = []

        language = inputs.iloc[:, 1][0]
        for tweet in inputs.iloc[:, 0]:
            if isinstance(tweet, list) and len(tweet) == 1:
                tweet = tweet[0]

            if 'en' in language:
                tweet_stemmed = [self._english_stemmer.stem(
                    word) for word in tweet]
            elif 'ar' in language:
                tweet_stemmed = [self._arabic_stemmer.stem(
                    word) for word in tweet]
            elif 'fr' in language:
                tweet_stemmed = [self._french_stemmer.stem(
                    word) for word in tweet]
            elif 'de' in language:
                tweet_stemmed = [self._german_stemmer.stem(
                    word) for word in tweet]
            elif 'es' in language:
                tweet_stemmed = [self._spanish_stemmer.stem(
                    word) for word in tweet]
            elif 'da' in language:
                tweet_stemmed = [self._danish_stemmer.stem(
                    word) for word in tweet]
            elif 'nl' in language:
                tweet_stemmed = [self._dutch_stemmer.stem(
                    word) for word in tweet]
            elif 'hu' in language:
                tweet_stemmed = [self._hungarian_stemmer.stem(
                    word) for word in tweet]
            elif 'it' in language:
                tweet_stemmed = [self._italian_stemmer.stem(
                    word) for word in tweet]
            elif 'no' in language:
                tweet_stemmed = [self._norwegian_stemmer.stem(
                    word) for word in tweet]
            elif 'po' in language:
                tweet_stemmed = [self._portuguese_stemmer.stem(
                    word) for word in tweet]
            elif 'ro' in language:
                tweet_stemmed = [self._romanian_stemmer.stem(
                    word) for word in tweet]
            elif 'ru' in language:
                tweet_stemmed = [self._russian_stemmer.stem(
                    word) for word in tweet]
            elif 'sv' in language:
                tweet_stemmed = [self._swedish_stemmer.stem(
                    word) for word in tweet]
            else:
                tweet_stemmed = tweet

            tweets_stemmed.append(tweet_stemmed)
        return tweets_stemmed
