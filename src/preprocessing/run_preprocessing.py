#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps
"""

import argparse, csv, pickle
from numpy.core.numeric import NaN
import pandas as pd
from sklearn.pipeline import make_pipeline
from src.preprocessing.preprocessors.column_dropper import ColumnDropper
from src.preprocessing.preprocessors.non_english_remover import NonEnglishRemover
from src.preprocessing.punctuation_remover import PunctuationRemover
from src.preprocessing.tokenizer import Tokenizer
from src.util import COLUMN_MENTIONS, COLUMN_TWEET, SUFFIX_TOKENIZED


def main():
    # setting up CLI
    parser = argparse.ArgumentParser(description = "Various preprocessing steps")
    parser.add_argument("input_file", help = "path to the input csv file")
    parser.add_argument("output_file", help = "path to the output csv file")
    parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
    parser.add_argument("-t", "--tokenize", action = "store_true", help = "tokenize given column into individual words")
    parser.add_argument("-o", "--other", action = "store_true", help = "remove non-english tweets and unnecessary columns")
    parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TWEET)
    parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
    args = parser.parse_args()

    # load data
    df = pd.read_csv(args.input_file,
                     quoting=csv.QUOTE_NONNUMERIC,
                     lineterminator="\n",
                     verbose=False,
                     dtype={"quote_url": object, "place": object, "tweet": object, "language": object, "thumbnail": object},
                     converters={'mentions': eval, 'photos': eval, 'urls': eval})

    # collect all preprocessors
    preprocessors = []
    if args.punctuation:
        preprocessors.append(PunctuationRemover())
    if args.tokenize:
        preprocessors.append(Tokenizer(args.tokenize_input, args.tokenize_input + SUFFIX_TOKENIZED))
    if args.other:   
        DROP_COLS = [            
            "id", "conversation_id", "created_at", "timezone", "user_id", "name", "place",
            "replies_count", "retweets_count", "likes_count", "language",
            # "cashtag" only few records have this filled. Might be useless
            # below columns have always the same value for all records
            "retweet", "near", "geo", "source", "user_rt_id", "user_rt", "retweet_id",
            "retweet_date", "translate", "trans_src", 'trans_dest\r']

        preprocessors.append(NonEnglishRemover())
        preprocessors.append(ColumnDropper(DROP_COLS))

    # call all preprocessing steps
    for preprocessor in preprocessors:
        df = preprocessor.fit_transform(df)

    # store the results
    df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

    # create a pipeline if necessary and store it as pickle file
    if args.export_file is not None:
        pipeline = make_pipeline(*preprocessors)
        with open(args.export_file, 'wb') as f_out:
            pickle.dump(pipeline, f_out)


if __name__ == "__main__":
    main()
